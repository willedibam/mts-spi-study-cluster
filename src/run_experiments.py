from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from hashlib import blake2s, sha256
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import zscore
import pandas as pd

from . import generators as generate
from .generators import generate_sin_mts_smooth, generate_lagged_mts, generate_lagged_warping_mts, generate_filter_roll_mts
from .generators.chat import generate_var_chat_a, generate_var_chat_b, generate_var_chat_c, generate_var_chat_d
from .compute import run_pyspi
from .mapping import DatasetMapping, ExperimentConfig
from .plot_style import apply_plot_style, save_figure
from .utils import dump_json, ensure_dir, load_json, project_root, slugify, timestamp, to_relative

import inspect


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def _pyspi_version() -> Dict[str, str]:
    """pyspi identity, recorded per dataset.

    pyspi 3.0 changed the *values* of ~70 SPIs relative to 2.x (kernel and
    symbolic estimators moved from bits to nats, the kraskov input policy
    changed, dcoh/xcorr/gd/ccm/coint_aeg/psi_wavelet/phase were corrected, and
    six directed spectral SPIs are transposed). Runs from different versions
    must not be pooled into one feature matrix, and nothing else in a dataset
    directory records which one produced it.

    Two fields, because they can disagree. `dist` is the installed
    distribution's metadata, which goes stale on an editable install whose
    version was bumped without reinstalling. `computation` is pyspi's own
    COMPUTATION_VERSION, read from the live source: it is what pyspi bumps
    whenever a change can alter a valid SPI output, and what its checkpoint
    manifests are keyed on, so it is the field to compare before pooling.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        dist = version("pyspi")
    except PackageNotFoundError:  # editable install without metadata
        dist = "unknown"
    try:
        from pyspi._parallel import COMPUTATION_VERSION as computation
    except Exception:
        computation = "unknown"
    return {"dist": dist, "computation": str(computation)}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_provenance() -> Dict[str, Any]:
    """Best-effort identity for the generator/runner source tree."""
    root = project_root()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PySPI experiments for a single dataset specification."
    )
    parser.add_argument(
        "--job-index",
        type=int,
        help="1-based dataset index (e.g. PBS_ARRAY_INDEX).",
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to an experiment YAML file.",
    )
    parser.add_argument(
        "--pyspi-config",
        help="Override PySPI config path.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Deprecated alias for --n-jobs (still sets PYSPI_N_JOBS env var). BLAS threads must be pinned to 1 in the shell.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        dest="n_jobs",
        help="PySPI worker process count. Passed directly to Calculator.compute(n_jobs=...). Overrides --threads.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        dest="checkpoint_dir",
        help="If set, per-SPI .npy checkpoints are written under <dataset_dir>/<checkpoint_dir>/. Enables resume across runs.",
    )
    parser.add_argument(
        "--mp-context",
        choices=["spawn", "fork", "forkserver"],
        dest="mp_context",
        help="Multiprocessing start method for PySPI workers (default: spawn).",
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
        "--heatmap",
        dest="heatmap",
        action="store_true",
        help="Generate mts_heatmap.png, overriding the experiment config.",
    )
    parser.add_argument(
        "--no-heatmap",
        dest="heatmap",
        action="store_false",
        help="Disable heatmap generation.",
    )
    parser.set_defaults(heatmap=None)
    parser.add_argument(
        "--no-csv",
        dest="csv",
        action="store_false",
        help="Skip calc.csv. The Calculator table is ~0.6 MB/dataset (parquet "
             "~2.7 MB) versus ~0.2 MB for spi_mpis.npz, which is the only one "
             "the downstream GNN pipeline reads. Use for runs whose sole "
             "consumer is training.",
    )
    parser.set_defaults(csv=True)
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Also export calc.parquet alongside calc.csv.",
    )
    parser.add_argument(
        "--mts-only",
        action="store_true",
        help="Generate timeseries.npy only (skip PySPI/heatmaps). Composes with --job-index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which dataset would run without executing generation or PySPI.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have meta.json, calc.csv, and spi_mpis.npz.",
    )
    parser.add_argument(
        "--regenerate-timeseries",
        action="store_true",
        help="Force regeneration of timeseries even if timeseries.npy exists.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.experiment_config)
    config = ExperimentConfig.from_file(config_path)
    if args.pyspi_config:
        config.pyspi_config = _resolve_path(args.pyspi_config)
    if args.normalise is not None:
        config.normalise = bool(args.normalise)
    if args.threads:
        config.threads = args.threads
    if config.timestamp:
        _run_ts = datetime.now().strftime("%y%m%d_%H%M%S")
        config.base_output_dir = config.base_output_dir.parent / (
            config.base_output_dir.name + f"_{_run_ts}"
        )
    mapping = DatasetMapping(config)
    experiment_provenance = {
        "config": to_relative(config_path),
        "config_sha256": _file_sha256(config_path),
        **_repository_provenance(),
    }
    if args.list:
        print(f"[INFO] Listing {len(mapping)} dataset combinations from {to_relative(config_path)}.")
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
        specs = list(mapping.specs)
    else:
        specs = [mapping.spec_for_index(args.job_index)]

    for spec in specs:
        print(f"[INFO] Running dataset {spec.index}/{len(mapping)}: {spec.name}")
        if args.dry_run:
            print(_describe_dataset(spec))
            continue
        if args.skip_existing and _dataset_complete(spec):
            print(
                f"[INFO] Skipping dataset {spec.name} "
                f"(found meta.json, calc.csv, and spi_mpis.npz in {to_relative(spec.dataset_dir)})."
            )
            continue
        # Explicit --n-jobs wins; otherwise --threads (legacy) or spec.threads
        # sets PYSPI_N_JOBS as a fallback for the kwarg default in compute().
        _configure_threading(args.threads or spec.threads)
        effective_n_jobs = args.n_jobs if args.n_jobs is not None else (args.threads or spec.threads)
        data, ts_path, gen_extras = _ensure_timeseries(spec, regenerate=args.regenerate_timeseries)
        if args.mts_only:
            if _heatmap_enabled(args.heatmap, spec.save_heatmap):
                save_mts_heatmap(data, ts_path.parent / "mts_heatmap.png")
            continue

        data = data.astype(np.float64, copy=False)
        dataset_dir = ts_path.parent
        cp_dir = (dataset_dir / args.checkpoint_dir) if args.checkpoint_dir else None
        compute_start = time.perf_counter()
        result = run_pyspi(
            data,
            config_path=spec.pyspi_config,
            normalise=spec.normalise,
            n_jobs=effective_n_jobs,
            checkpoint_dir=cp_dir,
            mp_context=args.mp_context,
        )
        compute_seconds = time.perf_counter() - compute_start
        csv_path = dataset_dir / "calc.csv"
        if args.csv:
            result.table.to_csv(csv_path, index=True)
        if args.parquet:
            _safe_write_parquet(result.table, dataset_dir / "calc.parquet")
        npz_path = dataset_dir / "spi_mpis.npz"
        np.savez_compressed(npz_path, **result.matrices)
        heatmap_paths: list[str] = []
        if _heatmap_enabled(args.heatmap, spec.save_heatmap):
            deltas = [max(1, int(d)) for d in (spec.heatmap_deltas or [1])]
            base_filename = "mts_heatmap.png"
            base_path = dataset_dir / base_filename
            save_mts_heatmap(data, base_path)
            heatmap_paths.append(base_filename)
            for delta in deltas:
                if delta == 1:
                    continue
                filename = f"mts_heatmap_delta{delta}.png"
                figure_path = dataset_dir / filename
                view = data[::delta]
                save_mts_heatmap(view, figure_path)
                heatmap_paths.append(filename)
        meta = _build_metadata(
            spec=spec,
            result=result,
            paths={
                "timeseries": "",
                "calc_csv": "calc.csv" if args.csv else "",
                "calc_parquet": "calc.parquet" if args.parquet else "",
                "spi_archive": "spi_mpis.npz",
                "per_spi": {
                    name: round(float(t), 6)
                    for name, t in (result.timings or {}).items()
                },
                "heatmaps": heatmap_paths,
            },
            compute_seconds=compute_seconds,
            gen_extras=gen_extras,
            experiment_provenance=experiment_provenance,
        )
        dump_json(dataset_dir / "meta.json", meta)
        print(
            f"[INFO] Stored SPI results in {to_relative(csv_path.parent)} "
            f"({len(result.metadata)} SPIs, {compute_seconds:.1f}s)."
        )


def _configure_threading(threads: int | None) -> None:
    """
    Set PYSPI_N_JOBS (parallel-SPI worker count) for pyspi-fork.

    BLAS threading (OMP/OPENBLAS/MKL) must be pinned to 1 *before* Python starts
    — i.e. in the PBS/shell script — because libopenblas reads its env vars at
    import time. Setting them here would be a no-op for the parent process, and
    forked workers inherit the live BLAS state in-memory, not fresh env vars.
    """
    if threads and threads > 0:
        os.environ["PYSPI_N_JOBS"] = str(threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(var) not in ("1", None):
            print(
                f"[WARN] {var}={os.environ.get(var)} — expected 1 to avoid BLAS oversubscription "
                f"when PYSPI_N_JOBS>1. Set it in your shell/PBS script before launching python."
            )


def _heatmap_enabled(cli_value: bool | None, config_value: bool) -> bool:
    """Resolve the explicit CLI override against the dataset configuration."""

    return bool(config_value if cli_value is None else cli_value)


def _safe_write_parquet(table: pd.DataFrame, path: Path) -> None:
    try:
        table.to_parquet(path, index=True)
        print(f"[INFO] Wrote {to_relative(path)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Skipped parquet export ({exc}).")


def _ensure_timeseries(spec, regenerate: bool) -> tuple[np.ndarray, Path, dict]:
    # If timeseries already exists on disk, reuse it to avoid reloading/downloading.
    if spec.source in {"real", "yfinance"} and not regenerate:
        existing = _load_existing_timeseries(spec)
        if existing:
            data, ts_path = existing
            print(f"[INFO] Loaded cached timeseries: {to_relative(ts_path)}")
            return data, ts_path, {}

    if spec.source == "real":
        data, M, T, dataset_slug, chosen_idx, channels_first = _load_real_sample(spec)
        spec.M = M
        spec.T = T
        spec.dataset_slug = dataset_slug
        spec.sample_index = chosen_idx
        spec.channels_first = channels_first
        spec.dataset_dir = ensure_dir(spec.base_output_dir / spec.class_dir / dataset_slug)
        dataset_dir = spec.dataset_dir
        ts_path = dataset_dir / "timeseries.npy"
        np.save(ts_path, data.astype(np.float32))
        print(
            f"[INFO] Loaded real dataset '{spec.dataset_name}' class '{spec.class_label}' "
            f"sample {chosen_idx} -> {to_relative(ts_path)} (shape {data.shape[0]}x{data.shape[1]})"
        )
        return data, ts_path, {}
    if spec.source == "yfinance":
        data, M, T, dataset_slug, tickers = _load_yfinance_sample(spec)
        spec.M = M
        spec.T = T
        spec.dataset_slug = dataset_slug
        spec.dataset_dir = ensure_dir(spec.base_output_dir / spec.class_dir / dataset_slug)
        dataset_dir = spec.dataset_dir
        ts_path = dataset_dir / "timeseries.npy"
        np.save(ts_path, data.astype(np.float32))
        print(
            f"[INFO] Loaded yfinance data for {tickers} ({spec.period}, {spec.interval}) "
            f"-> {to_relative(ts_path)} (shape {data.shape[0]}x{data.shape[1]})"
        )
        return data, ts_path, {}

    dataset_dir = ensure_dir(spec.dataset_dir)
    ts_path = dataset_dir / "timeseries.npy"
    full_path = dataset_dir / "full_lattice.npy"
    ground_truth_path = dataset_dir / "ground_truth.npz"
    wants_full_lattice = bool(
        spec.generator == "cml_logistic"
        and spec.generator_params.get("return_full_lattice", False)
    )
    wants_ground_truth = spec.generator in {
        "desai_zwanzig",
        "kuramoto_order_parameter",
        "miller_huse",
        "quadratic_cml_order_parameter",
        "stuart_landau",
    }
    gen_extras: dict = {}
    if (
        ts_path.exists()
        and not regenerate
        and (not wants_full_lattice or full_path.exists())
        and (not wants_ground_truth or ground_truth_path.exists())
    ):
        data = np.load(ts_path).astype(np.float64, copy=False)
        if wants_full_lattice:
            full_shape = list(np.load(full_path, mmap_mode="r").shape)
            gen_extras["full_lattice"] = {"path": full_path.name, "shape": full_shape}
        if wants_ground_truth:
            gen_extras["ground_truth"] = _ground_truth_descriptor(ground_truth_path)
        print(f"[INFO] Loaded cached timeseries: {to_relative(ts_path)}")
    else:
        start = time.perf_counter()
        data, gen_extras = generate_synthetic_from_spec(spec)
        np.save(ts_path, data.astype(np.float32))
        mother = gen_extras.pop("_mother", None)   # not JSON-serialisable -> save alongside, keep out of meta
        if mother is not None:
            np.save(dataset_dir / "mother.npy", np.asarray(mother, dtype=np.float32))
        full_lattice = gen_extras.pop("_full_lattice", None)
        if full_lattice is not None:
            full_array = np.asarray(full_lattice, dtype=np.float32)
            np.save(full_path, full_array)
            gen_extras["full_lattice"] = {
                "path": full_path.name,
                "shape": list(full_array.shape),
            }
        ground_truth = gen_extras.pop("_ground_truth", None)
        if ground_truth is not None:
            arrays = {name: np.asarray(values) for name, values in ground_truth.items()}
            np.savez_compressed(ground_truth_path, **arrays)
            gen_extras["ground_truth"] = _ground_truth_descriptor(ground_truth_path)
        duration = time.perf_counter() - start
        print(
            f"[INFO] Generated timeseries ({data.shape[0]}x{data.shape[1]}) "
            f"in {duration:.2f}s -> {to_relative(ts_path)}"
        )
    gen_extras.setdefault(
        "resolved_params",
        _resolve_generator_params(spec.generator, dict(spec.generator_params)),
    )
    return data.astype(np.float64, copy=False), ts_path, gen_extras


def _ground_truth_descriptor(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        descriptor: dict = {
            "path": path.name,
            "arrays": {name: list(archive[name].shape) for name in archive.files},
        }
        for name in (
            "r_full",
            "r_observed",
            "r_unobserved",
            "r_full_future",
            "r_unobserved_future",
            "magnetization",
            "spin_magnetization",
            "spin_magnetization_unobserved",
            "magnetization_future",
            "spin_magnetization_future",
            "spin_magnetization_unobserved_future",
            "magnetization_unobserved",
            "magnetization_unobserved_future",
            "order_parameter_R",
            "order_parameter_R_future",
            "mean_activity",
            "mean_activity_future",
        ):
            if name in archive.files:
                values = np.asarray(archive[name], dtype=np.float64)
                if values.size == 0:
                    continue
                descriptor[f"{name}_mean"] = float(values.mean())
                descriptor[f"{name}_std"] = float(values.std())
        for name in (
            "critical_coupling",
            "q_spin_rms",
            "q_spin_abs",
            "q_spin_rms_unobserved",
            "q_spin_abs_unobserved",
            "spin_m2",
            "spin_m4",
            "spin_binder_cumulant",
            "spin_susceptibility",
            "q_magnetization_rms",
            "q_magnetization_abs",
            "q_magnetization_rms_unobserved",
            "q_magnetization_abs_unobserved",
            "magnetization_m2",
            "magnetization_m4",
            "magnetization_binder_cumulant",
            "magnetization_susceptibility",
            "beta",
            "reduced_coupling",
            "exact_spontaneous_magnetization",
            "q_R_mean",
            "q_R_std",
            "q_activity_mean",
            "q_temporal_spectral_entropy",
            "q_dynamical_spatial_pattern_entropy",
            "q_selected_band_power",
            "q_period2_activity",
            "q_turbulent_fraction_0p05",
        ):
            if name in archive.files:
                value = float(archive[name])
                descriptor[name] = value if np.isfinite(value) else None
    return descriptor


def generate_synthetic_from_spec(spec) -> tuple[np.ndarray, dict]:
    """Generate synthetic MTS data for a spec without disk I/O or caching.

    Pure function: reads spec.generator, spec.M, spec.T, spec.rng_seed, and
    spec.generator_params, returns (data, gen_extras). Used by the CLI path
    (_ensure_timeseries wraps it with caching + np.save) and by notebooks
    that want in-memory parameter exploration.
    """
    generator_params = dict(spec.generator_params)
    gen_extras: dict = {}
    if spec.generator == "sin_mts_smooth":
        data, internals = generate_sin_mts_smooth(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            **generator_params,
        )
        gen_extras = {"a_values": internals.a_values.tolist()}
    elif spec.generator == "lagged_mts":
        data, internals = generate_lagged_mts(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            **generator_params,
        )
        gen_extras = {"lags": internals.lags.tolist()}
    elif spec.generator == "lagged_warping_mts":
        data, internals = generate_lagged_warping_mts(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            **generator_params,
        )
        gen_extras = {"lags": internals.lags.tolist()}
    elif spec.generator == "filter_roll_mts":
        data, internals = generate_filter_roll_mts(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            **generator_params,
        )
        gen_extras = {
            "types": internals.types,
            "betas": [None if np.isnan(b) else float(b) for b in internals.betas],
            "noise_stds": internals.noise_stds.tolist(),
            # transient: persisted to mother.npy by _ensure_timeseries, popped before meta.json
            "_mother": internals.mother,
        }
    elif spec.generator in ("var_chat_a", "var_chat_b", "var_chat_c", "var_chat_d"):
        _chat_gen = {
            "var_chat_a": generate_var_chat_a,
            "var_chat_b": generate_var_chat_b,
            "var_chat_c": generate_var_chat_c,
            "var_chat_d": generate_var_chat_d,
        }[spec.generator]
        data, internals = _chat_gen(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            **generator_params,
        )
        gen_extras = {
            "motif_node_indices": internals.motif_node_indices,
            "motif_edges": [list(e) for e in internals.motif_edges],
            "class_label": internals.class_label,
            "coupling_values": internals.coupling_values,
        }
    elif spec.generator == "cml_logistic" and generator_params.get("return_full_lattice", False):
        if generator_params.get("return_final_state", False):
            raise ValueError(
                "The experiment harness cannot persist both CML return_full_lattice "
                "and return_final_state; use direct generator calls for continuation."
            )
        generated = generate.generate_cml_logistic(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            **generator_params,
        )
        if generator_params.get("return_observation_indices", False):
            data, full_lattice, observation_indices = generated
            gen_extras = {
                "_full_lattice": full_lattice,
                "observation_indices": np.asarray(
                    observation_indices, dtype=np.int32
                ).tolist(),
            }
        else:
            data, full_lattice = generated
            gen_extras = {"_full_lattice": full_lattice}
    elif spec.generator == "kuramoto_order_parameter":
        generator_params.pop("return_internals", None)
        store_full_phases = bool(generator_params.pop("store_full_phases", True))
        data, internals = generate.generate_kuramoto_order_parameter(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_full_phases=store_full_phases,
            **generator_params,
        )
        ground_truth = {
            "r_full": internals.r_full.astype(np.float32),
            "r_observed": internals.r_observed.astype(np.float32),
            "r_unobserved": internals.r_unobserved.astype(np.float32),
            "frequencies": internals.frequencies.astype(np.float32),
            "observation_indices": internals.observation_indices.astype(np.int32),
            "sensor_offsets": internals.sensor_offsets.astype(np.float32),
            "initial_phases": internals.initial_phases.astype(np.float32),
            "final_phases": internals.final_phases.astype(np.float32),
            "critical_coupling": np.array(internals.critical_coupling),
        }
        if internals.full_phases is not None:
            ground_truth["full_phases"] = internals.full_phases.astype(np.float32)
        gen_extras = {
            "_ground_truth": ground_truth,
            "resolved_params": _resolve_generator_params(
                spec.generator,
                {**generator_params, "store_full_phases": store_full_phases},
            ),
        }
        if internals.r_full_future.size:
            gen_extras["_ground_truth"].update(
                {
                    "r_full_future": internals.r_full_future.astype(np.float32),
                    "r_unobserved_future": internals.r_unobserved_future.astype(np.float32),
                }
            )
    elif spec.generator == "miller_huse":
        generator_params.pop("return_internals", None)
        store_full_field = bool(generator_params.pop("store_full_field", False))
        persist_future_truth_series = bool(
            generator_params.pop("persist_future_truth_series", True)
        )
        data, internals = generate.generate_miller_huse(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_full_field=store_full_field,
            **generator_params,
        )
        primary = (
            internals.spin_magnetization_future
            if internals.spin_magnetization_future.size
            else internals.spin_magnetization
        )
        hidden = (
            internals.spin_magnetization_unobserved_future
            if internals.spin_magnetization_unobserved_future.size
            else internals.spin_magnetization_unobserved
        )
        second = float(np.mean(primary**2))
        fourth = float(np.mean(primary**4))
        binder = 1.0 - fourth / (3.0 * second**2) if second > 0.0 else np.nan
        susceptibility = internals.final_field.size * (
            second - float(np.mean(np.abs(primary))) ** 2
        )
        truth_block_count = min(8, len(primary))
        primary_blocks = np.asarray(
            [
                np.mean(np.abs(block))
                for block in np.array_split(primary, truth_block_count)
            ],
            dtype=np.float32,
        )
        hidden_blocks = np.asarray(
            [
                np.mean(np.abs(block))
                for block in np.array_split(hidden, truth_block_count)
            ],
            dtype=np.float32,
        )
        ground_truth = {
            "magnetization": internals.magnetization.astype(np.float32),
            "spin_magnetization": internals.spin_magnetization.astype(np.float32),
            "spin_magnetization_unobserved": (
                internals.spin_magnetization_unobserved.astype(np.float32)
            ),
            "q_spin_rms": np.array(np.sqrt(np.mean(primary**2)), dtype=np.float32),
            "q_spin_abs": np.array(np.mean(np.abs(primary)), dtype=np.float32),
            "q_spin_abs_blocks": primary_blocks,
            "q_spin_rms_unobserved": np.array(
                np.sqrt(np.mean(hidden**2)), dtype=np.float32
            ),
            "q_spin_abs_unobserved": np.array(
                np.mean(np.abs(hidden)), dtype=np.float32
            ),
            "q_spin_abs_unobserved_blocks": hidden_blocks,
            "spin_m2": np.array(second, dtype=np.float32),
            "spin_m4": np.array(fourth, dtype=np.float32),
            "spin_binder_cumulant": np.array(binder, dtype=np.float32),
            "spin_susceptibility": np.array(susceptibility, dtype=np.float32),
            "patch_indices": internals.patch_indices.astype(np.int32),
            "initial_field": internals.initial_field.astype(np.float32),
            "final_field": internals.final_field.astype(np.float32),
        }
        if persist_future_truth_series:
            ground_truth.update(
                {
                    "magnetization_future": internals.magnetization_future.astype(
                        np.float32
                    ),
                    "spin_magnetization_future": (
                        internals.spin_magnetization_future.astype(np.float32)
                    ),
                    "spin_magnetization_unobserved_future": (
                        internals.spin_magnetization_unobserved_future.astype(np.float32)
                    ),
                }
            )
        if internals.full_field is not None:
            ground_truth["full_field"] = internals.full_field.astype(np.float32)
        gen_extras = {
            "_ground_truth": ground_truth,
            "resolved_params": _resolve_generator_params(
                spec.generator,
                {
                    **generator_params,
                    "store_full_field": store_full_field,
                    "persist_future_truth_series": persist_future_truth_series,
                },
            ),
        }
    elif spec.generator == "desai_zwanzig":
        generator_params.pop("return_internals", None)
        store_full_states = bool(generator_params.pop("store_full_states", False))
        data, internals = generate.generate_desai_zwanzig(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_full_states=store_full_states,
            **generator_params,
        )
        primary = (
            internals.mean_field_future
            if internals.mean_field_future.size
            else internals.mean_field
        )
        truth_block_count = min(8, len(primary))
        ground_truth = {
            "mean_field": internals.mean_field.astype(np.float32),
            "mean_field_future": internals.mean_field_future.astype(np.float32),
            "q_mean_abs": np.array(np.mean(np.abs(primary)), dtype=np.float32),
            "q_mean_rms": np.array(np.sqrt(np.mean(primary**2)), dtype=np.float32),
            "q_mean_signed": np.array(np.mean(primary), dtype=np.float32),
            "q_mean_abs_blocks": np.asarray(
                [
                    np.mean(np.abs(block))
                    for block in np.array_split(primary, truth_block_count)
                ],
                dtype=np.float32,
            ),
            "observation_indices": internals.observation_indices.astype(np.int32),
            "initial_state": internals.initial_state.astype(np.float32),
            "final_state": internals.final_state.astype(np.float32),
            "reference_mean_field_sigma_c": np.array(
                generate.DESAI_ZWANZIG_REFERENCE_SIGMA_C, dtype=np.float32
            ),
        }
        if internals.full_states is not None:
            ground_truth["full_states"] = internals.full_states.astype(np.float32)
        gen_extras = {
            "_ground_truth": ground_truth,
            "resolved_params": _resolve_generator_params(
                spec.generator,
                {**generator_params, "store_full_states": store_full_states},
            ),
        }
    elif spec.generator == "stuart_landau":
        generator_params.pop("return_internals", None)
        store_full_states = bool(generator_params.pop("store_full_states", False))
        data, internals = generate.generate_stuart_landau(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_full_states=store_full_states,
            **generator_params,
        )
        primary_order = (
            internals.order_parameter_future
            if internals.order_parameter_future.size
            else internals.order_parameter
        )
        primary_activity = (
            internals.mean_activity_future
            if internals.mean_activity_future.size
            else internals.mean_activity
        )
        ground_truth = {
            "order_parameter_R": np.abs(internals.order_parameter).astype(np.float32),
            "order_parameter_phase": np.angle(internals.order_parameter).astype(np.float32),
            "mean_activity": internals.mean_activity.astype(np.float32),
            "order_parameter_R_future": np.abs(
                internals.order_parameter_future
            ).astype(np.float32),
            "order_parameter_phase_future": np.angle(
                internals.order_parameter_future
            ).astype(np.float32),
            "mean_activity_future": internals.mean_activity_future.astype(np.float32),
            "q_R_mean": np.array(np.mean(np.abs(primary_order)), dtype=np.float32),
            "q_R_std": np.array(np.std(np.abs(primary_order)), dtype=np.float32),
            "q_activity_mean": np.array(np.mean(primary_activity), dtype=np.float32),
            "frequencies": internals.frequencies.astype(np.float32),
            "observation_indices": internals.observation_indices.astype(np.int32),
            "initial_state_real": internals.initial_state.real.astype(np.float32),
            "initial_state_imag": internals.initial_state.imag.astype(np.float32),
            "final_state_real": internals.final_state.real.astype(np.float32),
            "final_state_imag": internals.final_state.imag.astype(np.float32),
        }
        if internals.full_states is not None:
            ground_truth["full_states_real"] = internals.full_states.real.astype(np.float32)
            ground_truth["full_states_imag"] = internals.full_states.imag.astype(np.float32)
        gen_extras = {
            "_ground_truth": ground_truth,
            "resolved_params": _resolve_generator_params(
                spec.generator,
                {**generator_params, "store_full_states": store_full_states},
            ),
        }
    elif spec.generator == "quadratic_cml_order_parameter":
        generator_params.pop("return_internals", None)
        store_truth_field = bool(generator_params.pop("store_truth_field", False))
        data, internals = generate.generate_quadratic_cml_order_parameter(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_truth_field=store_truth_field,
            **generator_params,
        )
        summary = internals.truth_summary
        turbulent_fraction = summary["turbulent_fraction"]
        ground_truth = {
            "q_temporal_spectral_entropy": np.array(
                summary["temporal_spectral_entropy"], dtype=np.float32
            ),
            "q_dynamical_spatial_pattern_entropy": np.array(
                summary["dynamical_spatial_pattern_entropy"], dtype=np.float32
            ),
            "q_selected_band_power": np.array(
                summary["selected_band_power"], dtype=np.float32
            ),
            "q_period2_activity": np.array(
                summary["period2_activity"], dtype=np.float32
            ),
            "q_turbulent_fraction_0p05": np.array(
                turbulent_fraction["0.05"], dtype=np.float32
            ),
            "q_regime_vector": np.array(
                [
                    summary["temporal_spectral_entropy"],
                    summary["dynamical_spatial_pattern_entropy"],
                    summary["selected_band_power"],
                    summary["period2_activity"],
                ],
                dtype=np.float32,
            ),
            "spatial_power_distribution": np.asarray(
                summary["spatial_power_distribution"], dtype=np.float32
            ),
            "spatial_correlation": np.asarray(
                summary["spatial_correlation"], dtype=np.float32
            ),
            "observation_indices": internals.observation_indices.astype(np.int32),
            "final_state": internals.final_state.astype(np.float32),
        }
        if internals.truth_field is not None:
            ground_truth["truth_field"] = internals.truth_field.astype(np.float32)
        gen_extras = {
            "_ground_truth": ground_truth,
            "resolved_params": _resolve_generator_params(
                spec.generator,
                {**generator_params, "store_truth_field": store_truth_field},
            ),
        }
    else:
        data = generate.generate_series(
            spec.generator,
            seed=spec.rng_seed,
            M=spec.M,
            T=spec.T,
            **generator_params,
        )
    gen_extras.setdefault("resolved_params", _resolve_generator_params(spec.generator, generator_params))
    return data, gen_extras


def _resolve_generator_params(generator: str | None, provided: dict) -> dict:
    """Full effective parameter set: generator-signature defaults overlaid with
    provided params. Records what actually ran, including unspecified defaults.
    Best-effort: returns just `provided` if the signature can't be introspected.
    """
    skip = {"M", "T", "rng", "seed", "return_internals", "return_final_state", "init_state"}
    try:
        fn = generate.GENERATOR_REGISTRY.get(generator)
        sig = inspect.signature(fn)
        resolved: dict = {}
        for name, p in sig.parameters.items():
            if name in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is not inspect._empty:
                resolved[name] = p.default
        resolved.update(provided)
        return resolved
    except (TypeError, ValueError):
        return dict(provided)


def _load_existing_timeseries(spec) -> tuple[np.ndarray, Path] | None:
    if spec.source == "real":
        class_dir = spec.base_output_dir / spec.class_dir
        cls_slug = slugify(str(spec.class_label))
        # New naming (prep_uea / prep_bciciv2a): class{slug}_I{n}
        # Old naming (_load_real_sample after knowing M/T): M{M}_T{T}_I{n}_class{slug}
        candidates = list(class_dir.glob(f"class{cls_slug}_I{spec.instance}")) + \
                     list(class_dir.glob(f"*_I{spec.instance}_class{cls_slug}"))
        for candidate in candidates:
            ts_path = candidate / "timeseries.npy"
            if not ts_path.exists():
                continue
            data = np.load(ts_path).astype(np.float64, copy=False)
            spec.dataset_dir = candidate
            spec.dataset_slug = candidate.name
            spec.M = data.shape[1]
            spec.T = data.shape[0]
            spec.channels_first = False
            return data, ts_path
    if spec.source == "yfinance":
        class_dir = spec.base_output_dir / spec.class_dir
        for candidate in class_dir.glob(f"*_I{spec.instance}"):
            ts_path = candidate / "timeseries.npy"
            if not ts_path.exists():
                continue
            data = np.load(ts_path).astype(np.float64, copy=False)
            if spec.m_assets and data.shape[1] != spec.m_assets:
                continue
            spec.dataset_dir = candidate
            spec.dataset_slug = candidate.name
            spec.M = data.shape[1]
            spec.T = data.shape[0]
            return data, ts_path
    return None


def _real_sample_seed(*, dataset_name: str, class_label: str, instance: int, base_seed: int) -> int:
    payload = f"{dataset_name}|{class_label}|{instance}|{base_seed}".encode("utf-8")
    digest = blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def _load_real_sample(spec) -> tuple[np.ndarray, int, int, str, int, bool]:
    if not spec.package or not spec.dataset_name or spec.class_label is None:
        raise ValueError("Real dataset spec missing package, dataset_name or class_label.")
    if spec.package.lower() == "aeon":
        try:
            from aeon.datasets import load_classification
        except ImportError as exc:  # noqa: BLE001
            raise ImportError("aeon is required for package='aeon'.") from exc
        X, y = load_classification(spec.dataset_name, split="train")
    elif spec.package.lower() == "sktime":
        try:
            from sktime.datasets import load_UCR_UEA_dataset
        except ImportError as exc:  # noqa: BLE001
            raise ImportError("sktime is required for package='sktime'.") from exc
        X, y = load_UCR_UEA_dataset(spec.dataset_name, split="train")
    else:
        raise ValueError(f"Unsupported package '{spec.package}'. Expected 'aeon' or 'sktime'.")

    y_arr = np.asarray(y)
    target_label = str(spec.class_label)
    mask = np.where(y_arr.astype(str) == target_label)[0]
    if mask.size == 0:
        raise ValueError(f"No samples found for class '{target_label}' in {spec.dataset_name}.")

    seed = _real_sample_seed(
        dataset_name=spec.dataset_name,
        class_label=target_label,
        instance=spec.instance,
        base_seed=spec.rng_seed,
    )
    rng = np.random.default_rng(seed)
    chosen_idx = int(rng.choice(mask))
    sample = np.asarray(X[chosen_idx], dtype=float)
    if sample.ndim == 1:
        sample = sample[None, :]
    channels_first = sample.shape[0] <= sample.shape[1]
    M = sample.shape[0] if channels_first else sample.shape[1]
    T = sample.shape[1] if channels_first else sample.shape[0]
    data = sample.T if channels_first else sample
    if spec.zscore_data:
        data = zscore(data, axis=0, nan_policy="omit")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    cls_slug = slugify(target_label)
    dataset_slug = f"M{M}_T{T}_I{spec.instance}_class{cls_slug}"
    return data, M, T, dataset_slug, chosen_idx, channels_first


def _static_market_tickers(market: str) -> list[str]:
    """
    Return a static snapshot of common index constituents.
    Used because yfinance 0.2.x removed tickers_sp500()/tickers_dow()/tickers_nasdaq
    and scraping endpoints are blocked on the cluster.
    """
    key = (
        market.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace("&", "and")
    )
    if key in {"sp500", "sandp500"}:
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "GOOG", "BRK-B", "TSLA", "AVGO",
            "JPM", "LLY", "UNH", "XOM", "V", "MA", "PG", "HD", "COST", "JNJ",
            "MRK", "ABBV", "CVX", "BAC", "KO", "CRM", "NFLX", "AMD", "PEP", "ADBE",
            "WMT", "TMO", "LIN", "MCD", "DIS", "ACN", "CSCO", "INTU", "ORCL", "ABT",
            "WFC", "QCOM", "CAT", "GE", "VZ", "IBM", "AMAT", "DHR", "INTC", "TXN",
            "UBER", "NOW", "PFE", "UNP", "LOW", "PM", "SPGI", "HON", "COP", "RTX",
            "AXP", "AMGN", "SYK", "ISRG", "NEE", "ELV", "GS", "PGR", "ETN", "T",
            "BKNG", "LRCX", "BLK", "MDT", "BSX", "TJX", "ADP", "VRTX", "C", "CI",
            "GILD", "MMC", "CB", "LMT", "SCHW", "PLD", "FI", "PANW", "TMUS", "DE",
        ]
    if key in {"dow", "djia", "dowjones", "dow30"}:
        return [
            "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS",
            "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK",
            "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT",
        ]
    if key in {"nasdaq", "nasdaq100", "ndx"}:
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "AVGO", "META", "TSLA", "GOOGL", "GOOG", "COST",
            "NFLX", "AMD", "ADBE", "PEP", "LIN", "CSCO", "TMUS", "INTU", "CMCSA", "QCOM",
            "INTC", "TXN", "AMAT", "HON", "AMGN", "ISRG", "BKNG", "LRCX", "VRTX", "GILD",
            "SBUX", "PANW", "MDLZ", "ADP", "MU", "ADI", "REGN", "MELI", "KLAC", "SNPS",
            "CDNS", "PYPL", "ASML", "MAR", "CSX", "ORLY", "MNST", "CTAS", "LULU", "NXPI",
            "PCAR", "ROST", "MRVL", "FTNT", "WDAY", "ODFL", "IDXX", "PAYX", "MCHP", "EXC",
            "KDP", "AEP", "CTSH", "EA", "AZN", "BIIB", "FAST", "XEL", "GEHC", "BKR",
            "CME", "DXCM", "TEAM", "SGEN", "ZS", "VRSK", "CPRT", "SIRI", "DLTR", "EBAY",
        ]
    return []


def _load_yfinance_sample(spec) -> tuple[np.ndarray, int, int, str, list[str]]:
    try:
        import yfinance as yf
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("yfinance is required for package='yfinance'.") from exc

    if not spec.m_assets:
        raise ValueError("m_assets (M) must be specified for yfinance sources.")
    period = spec.period or "1y"
    interval = spec.interval or "1d"

    universe: list[str] = []
    if spec.tickers:
        universe.extend(spec.tickers)
    if spec.market:
        market_tickers = _static_market_tickers(spec.market)
        if not market_tickers:
            raise RuntimeError(f"Unsupported or empty market '{spec.market}'.")
        universe.extend(market_tickers)
    universe = sorted({t.upper() for t in universe if t})
    if not universe:
        raise ValueError("No tickers available for yfinance source.")

    seed = _real_sample_seed(
        dataset_name=spec.market or "yfinance",
        class_label="tickers",
        instance=spec.instance,
        base_seed=spec.rng_seed,
    )
    rng = np.random.default_rng(seed)
    choices = rng.choice(universe, size=min(spec.m_assets, len(universe)), replace=False)
    data = yf.download(
        tickers=list(choices),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"yfinance returned no data for tickers {choices} ({period}, {interval}).")
    try:
        df_close = data.xs("Close", axis=1, level=1)
    except Exception:
        if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns:
            df_close = data["Close"]
        else:
            raise
    df_close = df_close.dropna()
    if df_close.empty:
        raise ValueError("No non-NaN close prices after dropna().")
    arr = df_close.to_numpy(dtype=float)
    T = arr.shape[0]
    M = arr.shape[1]
    if spec.zscore_data:
        arr = zscore(arr, axis=0, nan_policy="omit")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    tick_slug = "_".join(slugify(t) for t in choices)
    dataset_slug = f"{tick_slug}_{period}_{interval}_I{spec.instance}"
    return arr, M, T, dataset_slug, list(choices)


def build_mts_heatmap(data: np.ndarray):
    """Build an MTS heatmap Figure without any disk I/O.

    Leaves the current matplotlib backend untouched so notebooks keep their
    inline renderer. Constant pixels-per-sample scaling keeps heatmaps
    visually comparable across M/T combos (~1 px per timestep, ~24 px per
    channel at 300 dpi).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    T, M = data.shape
    fig_w = max(2.0, min(16.0, T / 300.0))
    fig_h = max(0.5, min(8.0, M / 12.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
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
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    return fig


def save_mts_heatmap(data: np.ndarray, path: Path) -> None:
    import matplotlib
    # Pin Agg for headless cluster runs; no-op if a backend is already set.
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig = build_mts_heatmap(data)
    out_path = Path(path).with_suffix(".png")
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"[INFO] Wrote heatmap to {to_relative(out_path)}")


def _kuramoto_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    distribution = str(params.get("frequency_distribution", "gaussian"))
    omega_std = float(params.get("omega_std", 1.0))
    critical = float(
        (gen_extras.get("ground_truth") or {}).get(
            "critical_coupling",
            generate.kuramoto_critical_coupling(distribution, omega_std),
        )
    )
    coupling = float(params.get("K", critical))
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "K",
            "value": coupling,
            "continuum_critical_value": critical,
            "reduced_name": "kappa",
            "reduced_value": coupling / critical,
        },
        "order_parameter": {
            "name": "Kuramoto phase coherence",
            "definition": "R_N(t)=abs(mean_j(exp(i*theta_j(t))))",
            "canonical_full_array": "r_full",
            "primary_analysis_array": "r_full_future" if future_truth else "r_full",
            "hidden_complement_sensitivity_array": (
                "r_unobserved_future" if future_truth else "r_unobserved"
            ),
            "observed_subset_diagnostic_array": "r_observed",
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


def _miller_huse_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "coupling",
            "value": float(params.get("coupling", 0.205)),
            "path_parameter": "mu",
            "path_value": float(params.get("mu", 3.0)),
        },
        "order_parameter": {
            "name": "Miller--Huse spin magnetization",
            "definition": "m_s(t)=mean_r(1[x_r(t)>=0]-1[x_r(t)<0])",
            "finite_system_scalar": "mean_t(abs(m_s(t)))",
            "primary_analysis_array": (
                "spin_magnetization_future"
                if future_truth and bool(params.get("persist_future_truth_series", True))
                else None
            ),
            "primary_scalar": "q_spin_abs",
            "primary_block_summary": "q_spin_abs_blocks",
            "hidden_complement_sensitivity_array": (
                "spin_magnetization_unobserved_future"
                if future_truth
                else "spin_magnetization_unobserved"
            ),
            "hidden_complement_scalar": "q_spin_abs_unobserved",
            "hidden_complement_block_summary": "q_spin_abs_unobserved_blocks",
            "rms_magnetization_sensitivity_scalar": "q_spin_rms",
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


def _desai_zwanzig_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "sigma",
            "value": float(params.get("sigma", 1.890)),
            "plane_parameter": "theta",
            "plane_value": float(params.get("theta", 4.0)),
        },
        "order_parameter": {
            "name": "Desai--Zwanzig first moment",
            "definition": "M1(t)=mean_i(x_i(t))",
            "finite_system_scalar": "mean_t(abs(M1(t)))",
            "primary_analysis_array": (
                "mean_field_future" if future_truth else "mean_field"
            ),
            "primary_scalar": "q_mean_abs",
            "primary_block_summary": "q_mean_abs_blocks",
            "signed_scalar": "q_mean_signed",
            "rms_sensitivity_scalar": "q_mean_rms",
            "mean_field_reference_sigma_c": 1.890,
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


def _stuart_landau_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    future_truth = int(params.get("future_truth_T", 0)) > 0
    return {
        "control": {
            "name": "frequency_half_width",
            "value": float(params.get("frequency_half_width", 0.8)),
            "plane_parameter": "coupling",
            "plane_value": float(params.get("coupling", 0.8)),
        },
        "order_parameter": {
            "name": "Stuart--Landau complex mean field",
            "definition": "Z(t)=mean_j(z_j(t)); R(t)=abs(Z(t))",
            "primary_analysis_array": (
                "order_parameter_R_future" if future_truth else "order_parameter_R"
            ),
            "primary_scalar": "q_R_mean",
            "variability_scalar": "q_R_std",
            "activity_scalar": "q_activity_mean",
            "future_truth_disjoint_from_input_window": future_truth,
            "included_in_timeseries_input": False,
        },
    }


def _quadratic_cml_semantics(spec, gen_extras: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(gen_extras.get("resolved_params") or spec.generator_params)
    return {
        "control": {
            "name": "alpha",
            "value": float(params.get("alpha", 1.8)),
            "path_parameter": "eps",
            "path_value": float(params.get("eps", 0.3)),
        },
        "order_parameter": {
            "name": "quadratic-CML physical regime-coordinate vector",
            "canonical_scalar_order_parameter": False,
            "definition": (
                "(temporal spectral entropy, dynamical spatial-pattern entropy, "
                "selected-band power, period-two residual) on a disjoint "
                "future full-lattice window"
            ),
            "primary_vector": "q_regime_vector",
            "scalar_diagnostics": [
                "q_temporal_spectral_entropy",
                "q_dynamical_spatial_pattern_entropy",
                "q_selected_band_power",
                "q_period2_activity",
            ],
            "future_truth_disjoint_from_input_window": True,
            "included_in_timeseries_input": False,
        },
    }


def _build_metadata(
    *,
    spec,
    result,
    paths: Dict[str, Any],
    compute_seconds: float,
    gen_extras: Dict[str, Any] | None = None,
    experiment_provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    variant_block = None
    if spec.variant:
        variant_block = {
            "name": spec.variant.name or "",
            "params": spec.variant.params,
            "slug": spec.variant.slug,
        }
    source_block: Dict[str, Any] = {"type": spec.source}
    if spec.source == "synthetic":
        extras = gen_extras or {}
        source_block.update(
            {
                "name": spec.generator,
                "params": spec.generator_params,
                "seed": spec.rng_seed,
                **extras,
            }
        )
        if spec.generator == "kuramoto_order_parameter":
            source_block.update(_kuramoto_semantics(spec, extras))
        elif spec.generator == "miller_huse":
            source_block.update(_miller_huse_semantics(spec, extras))
        elif spec.generator == "desai_zwanzig":
            source_block.update(_desai_zwanzig_semantics(spec, extras))
        elif spec.generator == "stuart_landau":
            source_block.update(_stuart_landau_semantics(spec, extras))
        elif spec.generator == "quadratic_cml_order_parameter":
            source_block.update(_quadratic_cml_semantics(spec, extras))
        # Fold in pre-seed provenance (e.g. adiabatic-continuation network_seed,
        # branch, dK) when the timeseries was produced outside the harness.
        prov_path = Path(spec.dataset_dir) / "gen_provenance.json"
        if prov_path.exists():
            try:
                source_block["preseed"] = load_json(prov_path)
            except (ValueError, OSError):
                pass
    elif spec.source == "real":
        source_block.update(
            {
                "package": spec.package,
                "dataset_name": spec.dataset_name,
                "class_label": spec.class_label,
                "sample_index": spec.sample_index,
                "zscore": spec.zscore_data,
            }
        )
    elif spec.source == "yfinance":
        source_block.update(
            {
                "package": spec.package,
                "tickers": spec.tickers,
                "market": spec.market,
                "period": spec.period,
                "interval": spec.interval,
                "M": spec.m_assets,
                "zscore": spec.zscore_data,
            }
        )
    return {
        "name": spec.name,
        "mts_class": spec.mts_class,
        "labels": spec.class_labels,
        "M": spec.M,
        "T": spec.T,
        "instance_index": spec.instance,
        "experiment": experiment_provenance or {},
        "sampling_design": {
            "seed_scope": spec.seed_scope,
            "seed_group_id": spec.seed_group_id,
            "role": (
                "paired-control-path"
                if "paired" in spec.class_labels
                else "independent-cell"
                if "independent-cell" in spec.class_labels
                else "unspecified"
            ),
        },
        "variant": variant_block,
        "normalise": spec.normalise,
        "timestamp": timestamp(),
        "generator": source_block,
        "pyspi": {
            "config": to_relative(spec.pyspi_config),
            "config_sha256": _file_sha256(spec.pyspi_config),
            "version": _pyspi_version(),
            "n_spis": len(result.metadata),
            "errors": result.errors or {},
            "spis": [
                {
                    "name": info.name,
                    "directed": info.directed,
                    "labels": info.labels,
                    "family": info.family,
                    "module": info.module,
                    "class_name": info.class_name,
                }
                for info in result.metadata
            ],
        },
        "paths": paths,
        "base_output_dir": to_relative(spec.base_output_dir),
        "dataset_dir": to_relative(spec.dataset_dir),
        "job": {
            "index": spec.index,
            "threads": spec.threads,
            "compute_seconds": compute_seconds,
        },
    }


def _dataset_complete(spec) -> bool:
    # calc.csv is deliberately NOT required: --no-csv runs are complete
    # without it, and requiring it would make --skip-existing recompute every
    # dataset on resume.
    dataset_dir = Path(spec.dataset_dir)
    required = [
        dataset_dir / "meta.json",
        dataset_dir / "spi_mpis.npz",
        dataset_dir / "timeseries.npy",
    ]
    if spec.generator in {
        "desai_zwanzig",
        "kuramoto_order_parameter",
        "miller_huse",
        "quadratic_cml_order_parameter",
        "stuart_landau",
    }:
        required.append(dataset_dir / "ground_truth.npz")
    if (
        spec.generator == "cml_logistic"
        and spec.generator_params.get("return_full_lattice", False)
    ):
        required.append(dataset_dir / "full_lattice.npy")
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
