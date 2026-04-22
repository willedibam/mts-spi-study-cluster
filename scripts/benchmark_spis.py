#!/usr/bin/env python
"""
Benchmark SPI computation times across an (M, T) grid.

Each (M, T) combo is run ``--repeats N`` times with the same seed; per-SPI and
total wall times are reported as mean/std across repeats so a single noisy run
does not dominate. Environment metadata (pyspi git sha, numpy version, python
version, platform, and a hash of installed versions of the core deps) is
captured at the top of the output file so results are pinnable to a specific
environment.

Usage:
    python scripts/benchmark_spis.py --pyspi-config configs/pyspi-v2/blended_config.yaml
    python scripts/benchmark_spis.py --preset headline
    python scripts/benchmark_spis.py --preset scaling --pyspi-config configs/pyspi-v2/fast_config.yaml
    python scripts/benchmark_spis.py --pyspi-config configs/pyspi/fast_config.yaml --repeats 5
    python scripts/benchmark_spis.py --resume

Presets:
    headline   two reference points: (M=10, T=500), (M=20, T=1000).
    scaling    cross-product of M={2,4,8,16,32,64} and T={100,200,400,800,1600,3200}.
    default    (no preset) use --M / --T arguments as a cross-product.

Results are saved incrementally after each (M, T) combo, so the run is
interrupt-safe. Use --resume to skip combos already computed at the same
``n_repeats``; combos present but with fewer repeats are recomputed.

Output schema:
    {
        "mts_class": "var1",
        "mts_class_labels": ["var", "linear", "ring-symmetric"],
        "generator": "varma",
        "generator_params": {...},
        "pyspi_config": "...",
        "family_filter": null,
        "n_repeats": 3,
        "seed": 42,
        "environment": {
            "pyspi_git_sha": "...", "pyspi_version": "...",
            "numpy_version": "...", "python_version": "...",
            "platform": "...",
            "dep_versions": {pkg: version, ...},
            "dep_fingerprint": "sha256:..."  # hash of dep_versions
        },
        "grid": {"M_values": [...], "T_values": [...], "points": [[M,T], ...] | null},
        "spi_metadata": {
            spi_name: {"family": ..., "module": ..., "class_name": ...,
                        "directed": bool, "labels": [...]},
            ...
        },
        "results": [
            {
                "mts_class": "var1", "M": 3, "T": 100, "n_repeats": 3,
                "total_seconds": {"mean": ..., "std": ..., "values": [...]},
                "peak_rss_mb_before": ..., "peak_rss_mb_after": ...,
                "n_spis": 120, "n_spis_failed": 0, "failed_spis": [],
                "timings": {
                    "SPI_name": {"mean": ..., "std": ..., "values": [...]},
                    ...
                }
            },
            ...
        ]
    }
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as im
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

# macOS: prevent crash when multiple OpenMP runtimes (numba, MKL, tslearn) are loaded
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

# Allow running as `python scripts/benchmark_spis.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.registry import generate_series
from src.compute import run_pyspi


DEFAULT_M = [3, 5, 10, 16, 20, 32, 64]
DEFAULT_T = [100, 500, 1000, 2000, 5000]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_REPEATS = 3
DEFAULT_MTS_CLASS = "var1"

PRESETS = {
    "headline": {
        "points": [(10, 500), (20, 1000)],
    },
    "scaling": {
        "M": [2, 4, 8, 16, 32, 64],
        "T": [100, 200, 400, 800, 1600, 3200],
    },
}

# MTS-class specs. Each entry pins a stable (name -> generator registry key +
# base_params + dataset-level labels). Keeping these in code rather than YAML so
# a benchmark run is self-contained: the JSON payload records exactly what was
# generated and how.
MTS_CLASS_SPECS = {
    "var1": {
        "generator": "varma",
        "labels": ["var", "linear", "ring-symmetric"],
        "base_params": {
            "phi": 0.8,
            "coupling": 0.4,
            "ma_phi": 0.0,
            "ma_coupling": 0.0,
            "noise_std": 0.1,
            "topology": "ring-symmetric",
            "transients": 100,
            "zscore": True,
        },
    },
    "kuramoto": {
        "generator": "kuramoto",
        "labels": ["kuramoto", "nonlinear", "oscillator", "all-to-all", "ODE"],
        "base_params": {
            "K": -4.0,
            "dt": 0.00625,
            "omega_mean": 3.0,
            "omega_std": 1.73205,  # 3.0 / sqrt(3)
            "eta": 0.0,
            "output": "sin",
            "connectivity": "all-to-all",
            "transients": 100,
            "zscore": True,
        },
    },
    "gaussian_noise": {
        "generator": "gaussian_noise",
        "labels": ["noise", None, "gaussian"],
        "base_params": {
            "zscore": True,
        },
    },
}

# Map config section prefixes to family names
FAMILY_SECTIONS = {
    "basic": ".statistics.basic",
    "distance": ".statistics.distance",
    "causal": ".statistics.causal",
    "infotheory": ".statistics.infotheory",
    "spectral": ".statistics.spectral",
    "wavelet": ".statistics.wavelet",
    "misc": ".statistics.misc",
}

# Core deps whose versions are recorded in the environment block. Anything not
# installed is silently omitted from dep_versions (and hence from the hash).
TRACKED_DEPS = (
    "pyspi", "numpy", "scipy", "pandas", "scikit-learn", "statsmodels",
    "mne", "mne-connectivity", "spectral-connectivity", "nitime",
    "hyppo", "cdt", "torch", "tslearn", "dtaidistance", "pyEDM",
    "h5py", "dill", "pyyaml",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pyspi-config", type=str, default="configs/pyspi-v2/blended_config.yaml",
                    help="PySPI config YAML to benchmark.")
    p.add_argument("--mts-class", type=str, default=DEFAULT_MTS_CLASS,
                    choices=list(MTS_CLASS_SPECS),
                    help="MTS dataset class to benchmark against.")
    p.add_argument("--preset", type=str, default=None, choices=list(PRESETS),
                    help="Grid preset; overrides --M/--T. 'headline' uses explicit (M,T) points.")
    p.add_argument("--M", type=int, nargs="+", default=DEFAULT_M,
                    help="Channel counts to benchmark (ignored if --preset is set).")
    p.add_argument("--T", type=int, nargs="+", default=DEFAULT_T,
                    help="Time lengths to benchmark (ignored if --preset is set).")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help=f"Repeats per (M,T) combo (default {DEFAULT_REPEATS}).")
    p.add_argument("--family", type=str, default=None,
                    choices=list(FAMILY_SECTIONS.keys()),
                    help="Only benchmark SPIs from this family.")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default: data/benchmark/timings_<class>_<config>.json).")
    p.add_argument("--resume", action="store_true",
                    help="Skip combos already computed at the same n_repeats.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for data generation.")
    return p.parse_args(argv)


def _resolve_grid(args) -> tuple[list[int], list[int], list[tuple[int, int]] | None]:
    """Return (M_values, T_values, explicit_points). If points is set, combos
    are exactly those points; otherwise the cross-product of M x T."""
    if args.preset is None:
        return args.M, args.T, None
    preset = PRESETS[args.preset]
    if "points" in preset:
        pts = preset["points"]
        Ms = sorted({p[0] for p in pts})
        Ts = sorted({p[1] for p in pts})
        return Ms, Ts, list(pts)
    return preset["M"], preset["T"], None


def _resolve_pyspi_config(config_str: str, family: str | None) -> Path:
    """If family is specified, create a temp config with only that family's SPIs."""
    config_path = PROJECT_ROOT / config_str
    if family is None:
        return config_path

    import yaml
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)

    section_prefix = FAMILY_SECTIONS[family]
    filtered = {k: v for k, v in full_cfg.items() if k.startswith(section_prefix)}
    if not filtered:
        raise ValueError(f"No SPIs found for family '{family}' (section prefix '{section_prefix}') in {config_path}")

    tmp_path = config_path.parent / f"_benchmark_{family}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(filtered, f, default_flow_style=False, sort_keys=False)
    return tmp_path


def _generate_data(mts_class: str, M: int, T: int, seed: int) -> np.ndarray:
    """Dispatch to the configured generator for ``mts_class`` with fixed params."""
    spec = MTS_CLASS_SPECS[mts_class]
    return generate_series(
        spec["generator"], seed=seed, M=M, T=T, **spec["base_params"]
    )


def _count_failed_spis(matrices: dict) -> tuple[int, list[str]]:
    """An SPI is considered failed if every off-diagonal cell is NaN.

    pyspi sets the whole column to NaN on exception (calc.table[spi] = nan),
    which reconstructs into an all-NaN matrix; a successful SPI has NaN
    diagonal only.
    """
    failed = []
    for name, m in matrices.items():
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
            continue
        off = m[~np.eye(m.shape[0], dtype=bool)]
        if off.size == 0:
            continue
        if np.all(np.isnan(off)):
            failed.append(name)
    return len(failed), sorted(failed)


def _peak_rss_mb() -> float:
    """Process peak RSS in MB. macOS reports bytes; Linux reports KB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return (r * scale) / (1024.0 * 1024.0)


def _pyspi_git_sha() -> str | None:
    """Resolve pyspi's install location and read the git sha if it's a git repo."""
    try:
        import pyspi
    except ImportError:
        return None
    repo = Path(pyspi.__file__).resolve().parent.parent
    if not (repo / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _dep_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_DEPS:
        try:
            out[name] = im.version(name)
        except im.PackageNotFoundError:
            continue
    return out


def _dep_fingerprint(versions: dict[str, str]) -> str:
    payload = ";".join(f"{k}=={v}" for k, v in sorted(versions.items()))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _build_environment(n_repeats: int) -> dict:
    versions = _dep_versions()
    return {
        "pyspi_git_sha": _pyspi_git_sha(),
        "pyspi_version": versions.get("pyspi"),
        "numpy_version": versions.get("numpy"),
        "python_version": platform.python_version(),
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "n_repeats": n_repeats,
        "dep_versions": versions,
        "dep_fingerprint": _dep_fingerprint(versions),
    }


def _load_existing(output_path: Path) -> dict | None:
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return None


def _save(output_path: Path, data: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(output_path)


def _completed_at_n_repeats(data: dict | None, n_repeats: int) -> set[tuple[int, int]]:
    """Combos already computed with at least ``n_repeats`` repeats."""
    if data is None:
        return set()
    return {
        (r["M"], r["T"]) for r in data.get("results", [])
        if r.get("n_repeats", 1) >= n_repeats and "error" not in r
    }


def _summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=0)), 6),
        "values": [round(float(v), 6) for v in arr],
    }


def _run_one_combo(mts_class: str, M: int, T: int, seed: int, pyspi_config: Path, repeats: int):
    """Run a single (M, T) combo ``repeats`` times. Returns (entry_dict, spi_metadata_or_None).

    Generator time is explicitly excluded from the timed region: ``t0`` starts
    after ``_generate_data`` returns. Per-SPI timings come from ``calc.timings``
    (perf_counter around each ``multivariate`` call in pyspi).
    """
    rss_before = _peak_rss_mb()
    totals: list[float] = []
    per_spi: dict[str, list[float]] = {}
    spi_metadata = None
    last_failed: list[str] = []
    last_n_failed = 0
    last_error = None

    for rep in range(repeats):
        data = _generate_data(mts_class, M, T, seed).astype(np.float64, copy=False)
        t0 = time.perf_counter()
        try:
            result = run_pyspi(
                data,
                config_path=pyspi_config,
                subset="default",
                normalise=False,
            )
            totals.append(time.perf_counter() - t0)
            timings = result.timings or {}
            for k, v in timings.items():
                per_spi.setdefault(k, []).append(float(v))
            if spi_metadata is None and result.metadata:
                spi_metadata = {
                    m.name: {
                        "family": m.family,
                        "module": m.module,
                        "class_name": m.class_name,
                        "directed": bool(m.directed),
                        "labels": list(m.labels),
                    }
                    for m in result.metadata
                }
            last_n_failed, last_failed = _count_failed_spis(result.matrices)
        except Exception as exc:
            totals.append(time.perf_counter() - t0)
            last_error = str(exc)
            break  # don't retry a failing combo

    rss_after = _peak_rss_mb()

    if last_error is not None:
        entry = {
            "mts_class": mts_class,
            "M": M, "T": T, "n_repeats": len(totals),
            "total_seconds": _summarise(totals) if totals else None,
            "n_spis": 0, "n_spis_failed": 0, "failed_spis": [],
            "timings": {},
            "peak_rss_mb_before": round(rss_before, 2),
            "peak_rss_mb_after": round(rss_after, 2),
            "error": last_error,
        }
        return entry, None

    timings_out = {k: _summarise(v) for k, v in per_spi.items()}
    entry = {
        "mts_class": mts_class,
        "M": M, "T": T, "n_repeats": repeats,
        "total_seconds": _summarise(totals),
        "n_spis": len(timings_out),
        "n_spis_failed": last_n_failed,
        "failed_spis": last_failed,
        "timings": timings_out,
        "peak_rss_mb_before": round(rss_before, 2),
        "peak_rss_mb_after": round(rss_after, 2),
    }
    return entry, spi_metadata


def _default_output_path(pyspi_config: str, mts_class: str, family: str | None) -> Path:
    """Derive a unique output path from config + mts-class + family."""
    cfg_stem = Path(pyspi_config).stem.replace("_config", "")
    suffix = f"_{family}" if family else ""
    return DEFAULT_OUTPUT_DIR / f"timings_{mts_class}_{cfg_stem}{suffix}.json"


def main(argv=None):
    args = parse_args(argv)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _default_output_path(args.pyspi_config, args.mts_class, args.family)

    pyspi_config = _resolve_pyspi_config(args.pyspi_config, args.family)
    M_values, T_values, explicit_points = _resolve_grid(args)

    if explicit_points is not None:
        combos = list(explicit_points)
    else:
        combos = [(M, T) for M in M_values for T in T_values]

    spec = MTS_CLASS_SPECS[args.mts_class]

    existing = _load_existing(output_path) if args.resume else None
    completed = _completed_at_n_repeats(existing, args.repeats)

    if existing is not None:
        # Resume guard: the existing file must be for the same mts_class.
        prev_class = existing.get("mts_class")
        if prev_class is not None and prev_class != args.mts_class:
            raise SystemExit(
                f"--resume: existing file {output_path} was generated for "
                f"mts_class={prev_class!r}, refusing to mix with {args.mts_class!r}."
            )
        output_data = existing
        output_data["environment"] = _build_environment(args.repeats)
        output_data["n_repeats"] = args.repeats
        output_data.setdefault("mts_class", args.mts_class)
        output_data.setdefault("mts_class_labels", spec["labels"])
        output_data.setdefault("generator", spec["generator"])
        output_data.setdefault("generator_params", spec["base_params"])
        output_data.setdefault("seed", args.seed)
    else:
        output_data = {
            "mts_class": args.mts_class,
            "mts_class_labels": spec["labels"],
            "generator": spec["generator"],
            "generator_params": spec["base_params"],
            "pyspi_config": args.pyspi_config,
            "family_filter": args.family,
            "n_repeats": args.repeats,
            "seed": args.seed,
            "environment": _build_environment(args.repeats),
            "grid": {
                "M_values": M_values, "T_values": T_values,
                "points": [list(p) for p in explicit_points] if explicit_points else None,
            },
            "results": [],
        }

    total = len(combos)
    skipped = 0

    for i, (M, T) in enumerate(combos, 1):
        if (M, T) in completed:
            skipped += 1
            print(f"[{i}/{total}] {args.mts_class} M={M}, T={T} — skipped (already have n_repeats>={args.repeats})")
            continue

        print(f"[{i}/{total}] {args.mts_class} M={M}, T={T} x {args.repeats} repeats ...", flush=True)
        t0 = time.perf_counter()
        entry, spi_metadata = _run_one_combo(args.mts_class, M, T, args.seed, pyspi_config, args.repeats)
        elapsed = time.perf_counter() - t0

        if "error" in entry:
            print(f"  ERROR after {elapsed:.1f}s: {entry['error']}")
        else:
            tot = entry["total_seconds"]
            print(f"  done in {elapsed:.1f}s wall "
                  f"(per-run mean {tot['mean']:.1f}s ± {tot['std']:.2f}s, "
                  f"{entry['n_spis']} SPIs, {entry['n_spis_failed']} failed, "
                  f"peak RSS {entry['peak_rss_mb_after']:.0f} MB)")

        if spi_metadata and "spi_metadata" not in output_data:
            output_data["spi_metadata"] = spi_metadata

        output_data["results"] = [
            r for r in output_data["results"] if (r["M"], r["T"]) != (M, T)
        ]
        output_data["results"].append(entry)
        _save(output_path, output_data)

    if skipped:
        print(f"\nDone. Skipped {skipped}/{total} combos (already computed at n_repeats>={args.repeats}).")
    else:
        print(f"\nDone. Computed {total} combos.")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
