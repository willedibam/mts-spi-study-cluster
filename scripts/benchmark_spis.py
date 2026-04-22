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
        "pyspi_config": "...",
        "family_filter": null,
        "n_repeats": 3,
        "environment": {
            "pyspi_git_sha": "...", "pyspi_version": "...",
            "numpy_version": "...", "python_version": "...",
            "platform": "...",
            "dep_versions": {pkg: version, ...},
            "dep_fingerprint": "sha256:..."  # hash of dep_versions
        },
        "grid": {"M_values": [...], "T_values": [...], "points": [[M,T], ...] | null},
        "spi_families": {spi_name: family, ...},
        "results": [
            {
                "M": 3, "T": 100, "n_repeats": 3,
                "total_seconds": {"mean": ..., "std": ..., "values": [...]},
                "peak_rss_mb_before": ..., "peak_rss_mb_after": ...,
                "n_spis": 120,
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

from src.generators.linear import generate_varma
from src.compute import run_pyspi


DEFAULT_M = [3, 5, 10, 16, 20, 32, 64]
DEFAULT_T = [100, 500, 1000, 2000, 5000]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmark" / "timings.json"
DEFAULT_REPEATS = 3

PRESETS = {
    "headline": {
        "points": [(10, 500), (20, 1000)],
    },
    "scaling": {
        "M": [2, 4, 8, 16, 32, 64],
        "T": [100, 200, 400, 800, 1600, 3200],
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
                    help="Output JSON path (default: data/benchmark/timings.json).")
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


def _generate_var1(M: int, T: int, seed: int) -> np.ndarray:
    """Generate a VAR(1) process for benchmarking."""
    return generate_varma(
        M=M, T=T,
        phi=0.6, coupling=0.3,
        ma_phi=0.0, ma_coupling=0.0,
        noise_std=0.2, transients=100,
        topology="ring-symmetric",
        rng=np.random.default_rng(seed),
        zscore=True,
    )


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


def _run_one_combo(M: int, T: int, seed: int, pyspi_config: Path, repeats: int):
    """Run a single (M, T) combo ``repeats`` times. Returns (entry_dict, spi_families_or_None)."""
    rss_before = _peak_rss_mb()
    totals: list[float] = []
    per_spi: dict[str, list[float]] = {}
    spi_families = None
    last_error = None

    for rep in range(repeats):
        data = _generate_var1(M, T, seed).astype(np.float64, copy=False)
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
            if spi_families is None and result.metadata:
                spi_families = {m.name: m.family for m in result.metadata}
        except Exception as exc:
            totals.append(time.perf_counter() - t0)
            last_error = str(exc)
            break  # don't retry a failing combo

    rss_after = _peak_rss_mb()

    if last_error is not None:
        entry = {
            "M": M, "T": T, "n_repeats": len(totals),
            "total_seconds": _summarise(totals) if totals else None,
            "n_spis": 0, "timings": {},
            "peak_rss_mb_before": round(rss_before, 2),
            "peak_rss_mb_after": round(rss_after, 2),
            "error": last_error,
        }
        return entry, None

    # Only keep SPIs that were timed on every repeat; if any repeat dropped
    # one, its summary would be on a smaller sample — flag by recording
    # the actual count.
    timings_out = {k: _summarise(v) for k, v in per_spi.items()}
    entry = {
        "M": M, "T": T, "n_repeats": repeats,
        "total_seconds": _summarise(totals),
        "n_spis": len(timings_out),
        "timings": timings_out,
        "peak_rss_mb_before": round(rss_before, 2),
        "peak_rss_mb_after": round(rss_after, 2),
    }
    return entry, spi_families


def main(argv=None):
    args = parse_args(argv)

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    if args.family and not args.output:
        output_path = output_path.with_name(f"timings_{args.family}.json")

    pyspi_config = _resolve_pyspi_config(args.pyspi_config, args.family)
    M_values, T_values, explicit_points = _resolve_grid(args)

    if explicit_points is not None:
        combos = list(explicit_points)
    else:
        combos = [(M, T) for M in M_values for T in T_values]

    existing = _load_existing(output_path) if args.resume else None
    completed = _completed_at_n_repeats(existing, args.repeats)

    if existing is not None:
        output_data = existing
        # Always refresh environment + n_repeats on a resume: old runs may have
        # been done against a different pyspi sha. We keep the old per-entry
        # rows (they include their own n_repeats).
        output_data["environment"] = _build_environment(args.repeats)
        output_data["n_repeats"] = args.repeats
    else:
        output_data = {
            "pyspi_config": args.pyspi_config,
            "family_filter": args.family,
            "n_repeats": args.repeats,
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
            print(f"[{i}/{total}] M={M}, T={T} — skipped (already have n_repeats>={args.repeats})")
            continue

        print(f"[{i}/{total}] M={M}, T={T} x {args.repeats} repeats ...", flush=True)
        t0 = time.perf_counter()
        entry, spi_families = _run_one_combo(M, T, args.seed, pyspi_config, args.repeats)
        elapsed = time.perf_counter() - t0

        if "error" in entry:
            print(f"  ERROR after {elapsed:.1f}s: {entry['error']}")
        else:
            tot = entry["total_seconds"]
            print(f"  done in {elapsed:.1f}s wall "
                  f"(per-run mean {tot['mean']:.1f}s ± {tot['std']:.2f}s, "
                  f"{entry['n_spis']} SPIs, peak RSS {entry['peak_rss_mb_after']:.0f} MB)")

        if spi_families and "spi_families" not in output_data:
            output_data["spi_families"] = spi_families

        # Replace any prior row for this (M,T) — the new one is strictly better.
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
