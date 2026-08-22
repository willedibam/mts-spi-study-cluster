from __future__ import annotations

import argparse
import os
from datetime import datetime
from hashlib import blake2s
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
            if args.heatmap or spec.save_heatmap:
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
        if args.heatmap or spec.save_heatmap:
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
        "kuramoto_order_parameter",
        "miller_huse",
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
            np.savez(ground_truth_path, **arrays)
            gen_extras["ground_truth"] = _ground_truth_descriptor(ground_truth_path)
        duration = time.perf_counter() - start
        print(
            f"[INFO] Generated timeseries ({data.shape[0]}x{data.shape[1]}) "
            f"in {duration:.2f}s -> {to_relative(ts_path)}"
        )
    return data.astype(np.float64, copy=False), ts_path, gen_extras


def _ground_truth_descriptor(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        descriptor: dict = {
            "path": path.name,
            "arrays": {name: list(archive[name].shape) for name in archive.files},
        }
        for name in ("r_full", "r_observed", "magnetization", "spin_magnetization"):
            if name in archive.files:
                values = np.asarray(archive[name], dtype=np.float64)
                descriptor[f"{name}_mean"] = float(values.mean())
                descriptor[f"{name}_std"] = float(values.std())
        if "critical_coupling" in archive.files:
            descriptor["critical_coupling"] = float(archive["critical_coupling"])
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
        data, full_lattice = generate.generate_cml_logistic(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            **generator_params,
        )
        gen_extras = {"_full_lattice": full_lattice}
    elif spec.generator == "kuramoto_order_parameter":
        generator_params.pop("return_internals", None)
        generator_params.pop("store_full_phases", None)
        data, internals = generate.generate_kuramoto_order_parameter(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_full_phases=True,
            **generator_params,
        )
        if internals.full_phases is None:  # defensive; forced above
            raise RuntimeError("Kuramoto full phases were not retained")
        gen_extras = {
            "_ground_truth": {
                "full_phases": internals.full_phases.astype(np.float32),
                "r_full": internals.r_full.astype(np.float32),
                "r_observed": internals.r_observed.astype(np.float32),
                "frequencies": internals.frequencies.astype(np.float32),
                "observation_indices": internals.observation_indices.astype(np.int32),
                "sensor_offsets": internals.sensor_offsets.astype(np.float32),
                "initial_phases": internals.initial_phases.astype(np.float32),
                "final_phases": internals.final_phases.astype(np.float32),
                "critical_coupling": np.array(internals.critical_coupling),
            }
        }
    elif spec.generator == "miller_huse":
        generator_params.pop("return_internals", None)
        generator_params.pop("store_full_field", None)
        data, internals = generate.generate_miller_huse(
            M=spec.M,
            T=spec.T,
            rng=np.random.default_rng(spec.rng_seed),
            return_internals=True,
            store_full_field=True,
            **generator_params,
        )
        if internals.full_field is None:  # defensive; forced above
            raise RuntimeError("Miller--Huse full field was not retained")
        gen_extras = {
            "_ground_truth": {
                "full_field": internals.full_field.astype(np.float32),
                "magnetization": internals.magnetization.astype(np.float32),
                "spin_magnetization": internals.spin_magnetization.astype(np.float32),
                "patch_indices": internals.patch_indices.astype(np.int32),
                "initial_field": internals.initial_field.astype(np.float32),
                "final_field": internals.final_field.astype(np.float32),
            }
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


def _build_metadata(
    *,
    spec,
    result,
    paths: Dict[str, Any],
    compute_seconds: float,
    gen_extras: Dict[str, Any] | None = None,
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
        source_block.update(
            {
                "name": spec.generator,
                "params": spec.generator_params,
                "seed": spec.rng_seed,
                **(gen_extras or {}),
            }
        )
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
        "variant": variant_block,
        "normalise": spec.normalise,
        "timestamp": timestamp(),
        "generator": source_block,
        "pyspi": {
            "config": to_relative(spec.pyspi_config),
            "version": _pyspi_version(),
            "n_spis": len(result.metadata),
            "errors": result.errors or {},
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
    if spec.generator in {"kuramoto_order_parameter", "miller_huse"}:
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
