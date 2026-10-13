from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import blake2s
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import zscore
import pandas as pd

from .generators import harness
from .generators.harness import generate_from_spec as generate_synthetic_from_spec
from .compute import run_pyspi
from .mapping import DatasetMapping, ExperimentConfig
from .plot_style import apply_plot_style, save_figure
from .utils import ensure_dir, load_json, project_root, slugify, timestamp, to_relative
from .provenance import (
    atomic_json, atomic_savez, file_sha256, peak_memory_mib, pyspi_version, repository_provenance,
    worker_resources,
)


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


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
        help="Older spelling of --n-jobs.",
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
        "config_sha256": file_sha256(config_path),
        **repository_provenance(),
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
        # --n-jobs wins; --threads and the config's `threads` are older spellings.
        effective_n_jobs = args.n_jobs or args.threads or spec.threads or 1
        resources = worker_resources(effective_n_jobs)
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
            random_seed=spec.rng_seed,
        )
        compute_seconds = time.perf_counter() - compute_start
        csv_path = dataset_dir / "calc.csv"
        if args.csv:
            result.table.to_csv(csv_path, index=True)
        if args.parquet:
            _safe_write_parquet(result.table, dataset_dir / "calc.parquet")
        npz_path = dataset_dir / "spi_mpis.npz"
        atomic_savez(npz_path, result.matrices)
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
        meta["job"].update(resources, n_jobs=effective_n_jobs, peak_memory_mib=peak_memory_mib())
        meta["pyspi"]["estimator_rng"] = {
            "policy": result.rng_policy,
            "seed": spec.rng_seed,
            "n_jobs": effective_n_jobs,
        }
        atomic_json(dataset_dir / "meta.json", meta)
        print(
            f"[INFO] Stored SPI results in {to_relative(csv_path.parent)} "
            f"({len(result.metadata)} SPIs, {compute_seconds:.1f}s)."
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
    expected = harness.expected_files(spec)
    wants_full_lattice = full_path.name in expected
    wants_ground_truth = ground_truth_path.name in expected
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
        # Persist exactly the input used by pyspi: a retry must not switch from
        # the generated float64 values to a rounded float32 realization.
        data = np.asarray(data, dtype=np.float64)
        np.save(ts_path, data)
        mother = gen_extras.pop("_mother", None)   # not JSON-serialisable -> save alongside, keep out of meta
        if mother is not None:
            np.save(dataset_dir / "mother.npy", np.asarray(mother, dtype=np.float64))
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
        harness.resolve_params(spec.generator, dict(spec.generator_params)),
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
        source_block.update(harness.semantics(spec, extras))
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
            "config_sha256": file_sha256(spec.pyspi_config),
            "version": pyspi_version(),
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
        dataset_dir / name
        for name in ("meta.json", "spi_mpis.npz", "timeseries.npy", *harness.expected_files(spec))
    ]
    if not all(path.exists() for path in required):
        return False
    # Finished, but by which pyspi? A resumed corpus must not mix computation
    # revisions or SPI configs, and recomputing only this job's indices would
    # mix them just the same, so a stale directory stops the run.
    recorded = load_json(dataset_dir / "meta.json").get("pyspi", {})
    recorded = (recorded.get("version", {}).get("computation"), recorded.get("config_sha256"))
    current = (pyspi_version()["computation"], file_sha256(spec.pyspi_config))
    if recorded != current:
        raise RuntimeError(
            f"{to_relative(dataset_dir)} was computed with pyspi {recorded[0]} "
            f"(config {str(recorded[1])[:12]}), not {current[0]} (config {current[1][:12]}). "
            "Use a new base_output_dir, or remove the old outputs to recompute the corpus."
        )
    return True


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
