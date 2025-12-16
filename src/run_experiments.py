from __future__ import annotations

import argparse
import ast
import os
from hashlib import blake2s
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import zscore
import pandas as pd

from . import generate
from .compute import run_pyspi
from .mapping import DatasetMapping, ExperimentConfig
from .plot_style import apply_plot_style, save_figure
from .utils import dump_json, ensure_dir, project_root, slugify, timestamp, to_relative


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
        "--mts-only",
        action="store_true",
        help="Generate timeseries.npy only (skip PySPI/heatmaps). If no job-index is given, runs all.",
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
    config_path = Path(args.experiment_config)
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
    if args.mts_only and args.job_index is None:
        specs = list(mapping.specs)
    else:
        if args.job_index is None:
            raise SystemExit("--job-index is required unless --list/--count-only/--mts-only is used.")
        specs = [mapping.spec_for_index(args.job_index)]

    for spec in specs:
        print(f"[INFO] Running dataset {spec.index}/{len(mapping)}: {spec.name}")
        if args.dry_run:
            print(_describe_dataset(spec))
            continue
        if args.skip_existing and _dataset_complete(spec.dataset_dir):
            print(
                f"[INFO] Skipping dataset {spec.name} "
                f"(found meta.json, calc.csv, and spi_mpis.npz in {to_relative(spec.dataset_dir)})."
            )
            continue
        _export_thread_hints(args.threads or spec.threads)
        data, ts_path = _ensure_timeseries(spec, regenerate=args.regenerate_timeseries)
        if args.mts_only:
            continue

        data = data.astype(np.float64, copy=False)
        dataset_dir = ts_path.parent
        compute_start = time.perf_counter()
        result = run_pyspi(
            data,
            config_path=spec.pyspi_config,
            subset=spec.pyspi_subset,
            normalise=spec.normalise,
        )
        compute_seconds = time.perf_counter() - compute_start
        csv_path = dataset_dir / "calc.csv"
        result.table.to_csv(csv_path, index=True)
        parquet_path = dataset_dir / "calc.parquet"
        if args.parquet:
            _safe_write_parquet(result.table, parquet_path)
        npz_path = dataset_dir / "spi_mpis.npz"
        np.savez_compressed(npz_path, **result.matrices)
        heatmap_required = args.heatmap or spec.save_heatmap
        heatmap_paths: list[str] = []
        if heatmap_required:
            deltas = [max(1, int(d)) for d in (spec.heatmap_deltas or [1])]
            base_filename = "mts_heatmap.png"
            base_path = dataset_dir / base_filename
            _save_heatmap(data, base_path)
            heatmap_paths.append(base_filename)
            for delta in deltas:
                if delta == 1:
                    continue
                filename = f"mts_heatmap_delta{delta}.png"
                figure_path = dataset_dir / filename
                view = data[::delta]
                _save_heatmap(view, figure_path)
                heatmap_paths.append(filename)
        meta = _build_metadata(
            spec=spec,
            result=result,
            paths={
                "timeseries": "",
                "calc_csv": "calc.csv",
                "calc_parquet": "calc.parquet" if args.parquet else "",
                "spi_archive": "spi_mpis.npz",
                "per_spi": {},
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


def _ensure_timeseries(spec, regenerate: bool) -> tuple[np.ndarray, Path]:
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
        return data, ts_path
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
        return data, ts_path

    dataset_dir = ensure_dir(spec.dataset_dir)
    ts_path = dataset_dir / "timeseries.npy"
    if ts_path.exists() and not regenerate:
        data = np.load(ts_path).astype(np.float64, copy=False)
        print(f"[INFO] Loaded cached timeseries: {to_relative(ts_path)}")
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
        np.save(ts_path, data.astype(np.float32))
        duration = time.perf_counter() - start
        print(
            f"[INFO] Generated timeseries ({data.shape[0]}x{data.shape[1]}) "
            f"in {duration:.2f}s -> {to_relative(ts_path)}"
        )
    return data.astype(np.float64, copy=False), ts_path


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
    source_block: Dict[str, Any] = {"type": spec.source}
    if spec.source == "synthetic":
        source_block.update(
            {
                "name": spec.generator,
                "params": spec.generator_params,
                "seed": spec.rng_seed,
            }
        )
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
        dataset_dir / "calc.csv",
        dataset_dir / "spi_mpis.npz",
        dataset_dir / "timeseries.npy",
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
