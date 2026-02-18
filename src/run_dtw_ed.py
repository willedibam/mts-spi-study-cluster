#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.stats import spearmanr

from .utils import dump_json, ensure_dir, project_root, to_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate warped AR(1) series per step size, compute Euclidean/DTW distances, "
            "and store per-step and overall correlation summaries."
        )
    )
    parser.add_argument(
        "--steps",
        required=True,
        help="Comma-separated step sizes, e.g. 1,2,4,8,16",
    )
    parser.add_argument(
        "--n-warps",
        type=int,
        default=64,
        help="Number of warped copies per step (default: 64).",
    )
    parser.add_argument("--T", type=int, default=1000, help="Series length (default: 1000).")
    parser.add_argument("--a", type=float, default=0.6, help="AR(1) coefficient (default: 0.6).")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="AR(1) noise std dev (default: 0.1).",
    )
    parser.add_argument("--p", type=float, default=0.5, help="Warp path direction probability.")
    parser.add_argument("--ar-seed", type=int, default=0, help="Seed for AR(1) generation.")
    parser.add_argument(
        "--warp-seed",
        type=int,
        default=42,
        help="Base seed used to derive per-step/per-replicate warp seeds.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Optional Sakoe-Chiba window for DTW (dtaidistance window parameter).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/dtw_ed",
        help="Output directory root (default: analysis/dtw_ed).",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run subdirectory name. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--allow-python-fallback",
        action="store_true",
        help="Allow Python DTW backend when C backend is unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run directory.",
    )
    return parser.parse_args()


def parse_steps(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No step values provided in --steps.")
    steps = [int(token) for token in tokens]
    if any(step <= 0 for step in steps):
        raise ValueError("All step values must be positive integers.")
    return steps


def generate_ar1(T: int, a: float, noise_std: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros(T, dtype=float)
    for t in range(1, T):
        X[t] = a * X[t - 1] + rng.normal(0.0, noise_std)
    return X


def warp_path(T: int, p: float = 0.5, step: int = 1, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = 0, 0
    path = [(x, y)]
    while x < T - 1 or y < T - 1:
        if x >= T - 1:
            y = min(y + step, T - 1)
        elif y >= T - 1:
            x = min(x + step, T - 1)
        else:
            if rng.random() < p:
                x = min(x + step, T - 1)
            else:
                y = min(y + step, T - 1)
        path.append((x, y))
    return np.asarray(path, dtype=int)


def path_to_mapping(path: np.ndarray, T: int) -> np.ndarray:
    n = np.zeros(T, dtype=int)
    s = np.zeros(T, dtype=float)
    for x, y in path:
        n[y] += 1
        s[y] += x
    mapping = np.zeros(T, dtype=int)
    for y in range(T):
        mapping[y] = int(round(s[y] / n[y])) if n[y] > 0 else 0
    mapping = np.clip(mapping, 0, T - 1)
    mapping = np.maximum.accumulate(mapping)
    return mapping


def warp_series(X_orig: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return X_orig[mapping].copy()


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    result = spearmanr(x, y)
    return float(result.statistic)


def resolve_output_root(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def infer_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    window_label = "none" if args.window is None else str(args.window)
    return f"dtw_ed_T{args.T}_warps{args.n_warps}_window{window_label}_{stamp}"


def check_c_backend() -> tuple[bool, str]:
    try:
        dtw._check_library(raise_exception=True)
        return True, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)


def iter_rows(steps: Iterable[int], n_warps: int):
    for step in steps:
        for replicate in range(n_warps):
            yield step, replicate


def main() -> None:
    args = parse_args()
    steps = parse_steps(args.steps)
    if args.n_warps <= 0:
        raise SystemExit("--n-warps must be > 0.")
    if args.T <= 1:
        raise SystemExit("--T must be > 1.")
    if args.window is not None and args.window <= 0:
        raise SystemExit("--window must be > 0 when provided.")

    c_backend_ok, c_backend_error = check_c_backend()
    if not c_backend_ok and not args.allow_python_fallback:
        raise SystemExit(
            "dtaidistance C backend unavailable.\n"
            f"Reason: {c_backend_error}\n"
            "Install/enable the C backend, or rerun with --allow-python-fallback."
        )

    output_root = resolve_output_root(args.output_dir)
    run_dir = output_root / infer_run_name(args)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Run directory already exists and is non-empty: {run_dir}\n"
            "Use --run-name to choose a new folder, or pass --overwrite."
        )
    ensure_dir(run_dir)

    X = generate_ar1(T=args.T, a=args.a, noise_std=args.noise_std, seed=args.ar_seed)

    n_steps = len(steps)
    euclidean_matrix = np.zeros((n_steps, args.n_warps), dtype=float)
    dtw_matrix = np.zeros((n_steps, args.n_warps), dtype=float)
    warp_seed_matrix = np.zeros((n_steps, args.n_warps), dtype=np.int64)

    rows: list[dict[str, float | int | str]] = []

    for step_idx, step in enumerate(steps):
        for _, replicate in iter_rows([step], args.n_warps):
            warp_seed = int(args.warp_seed + step * 1_000_000 + replicate)
            path = warp_path(T=args.T, p=args.p, step=step, seed=warp_seed)
            mapping = path_to_mapping(path, T=args.T)
            X_warp = warp_series(X, mapping)

            euclidean_dist = float(np.linalg.norm(X - X_warp))
            dtw_kwargs = {}
            if args.window is not None:
                dtw_kwargs["window"] = int(args.window)
            if c_backend_ok:
                dtw_dist = float(dtw.distance(X, X_warp, use_c=True, **dtw_kwargs))
                backend = "c"
            else:
                dtw_dist = float(dtw.distance(X, X_warp, use_c=False, **dtw_kwargs))
                backend = "python"

            euclidean_matrix[step_idx, replicate] = euclidean_dist
            dtw_matrix[step_idx, replicate] = dtw_dist
            warp_seed_matrix[step_idx, replicate] = warp_seed

            rows.append(
                {
                    "step": step,
                    "replicate": replicate,
                    "warp_seed": warp_seed,
                    "euclidean_distance": euclidean_dist,
                    "dtw_distance": dtw_dist,
                    "distance_ratio_dtw_over_euclidean": (
                        dtw_dist / euclidean_dist if euclidean_dist > 0 else np.nan
                    ),
                    "dtw_backend": backend,
                }
            )

    samples = pd.DataFrame(rows)

    summary_rows: list[dict[str, float | int]] = []
    for step in steps:
        g = samples.loc[samples["step"] == step]
        x = g["euclidean_distance"].to_numpy(dtype=float)
        y = g["dtw_distance"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "step": int(step),
                "n_samples": int(g.shape[0]),
                "euclidean_mean": float(np.mean(x)),
                "euclidean_std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
                "dtw_mean": float(np.mean(y)),
                "dtw_std": float(np.std(y, ddof=1)) if y.size > 1 else float("nan"),
                "pearson_r": safe_pearson(x, y),
                "spearman_rho": safe_spearman(x, y),
            }
        )
    summary_by_step = pd.DataFrame(summary_rows)

    x_all = samples["euclidean_distance"].to_numpy(dtype=float)
    y_all = samples["dtw_distance"].to_numpy(dtype=float)
    overall_summary = {
        "n_total_samples": int(samples.shape[0]),
        "steps": steps,
        "n_warps_per_step": int(args.n_warps),
        "pearson_r": safe_pearson(x_all, y_all),
        "spearman_rho": safe_spearman(x_all, y_all),
    }

    samples.to_parquet(run_dir / "samples.parquet", index=False)
    samples.to_csv(run_dir / "samples.csv", index=False)
    summary_by_step.to_parquet(run_dir / "summary_by_step.parquet", index=False)
    summary_by_step.to_csv(run_dir / "summary_by_step.csv", index=False)
    np.savez_compressed(
        run_dir / "distance_arrays.npz",
        steps=np.asarray(steps, dtype=int),
        warp_seed_matrix=warp_seed_matrix,
        euclidean_matrix=euclidean_matrix,
        dtw_matrix=dtw_matrix,
    )

    run_meta = {
        "run_dir": str(run_dir),
        "params": {
            "steps": steps,
            "n_warps": args.n_warps,
            "T": args.T,
            "a": args.a,
            "noise_std": args.noise_std,
            "p": args.p,
            "ar_seed": args.ar_seed,
            "warp_seed_base": args.warp_seed,
            "window": args.window,
        },
        "dtw_backend": {
            "c_available": bool(c_backend_ok),
            "allow_python_fallback": bool(args.allow_python_fallback),
            "c_backend_error": c_backend_error,
        },
        "overall_summary": overall_summary,
        "files": {
            "samples_parquet": "samples.parquet",
            "samples_csv": "samples.csv",
            "summary_by_step_parquet": "summary_by_step.parquet",
            "summary_by_step_csv": "summary_by_step.csv",
            "distance_arrays": "distance_arrays.npz",
        },
    }
    dump_json(run_dir / "run_meta.json", run_meta)

    print(f"[INFO] Output directory: {to_relative(run_dir)}")
    print(f"[INFO] DTW backend used: {'C' if c_backend_ok else 'Python'}")
    print(
        "[INFO] Overall correlation: "
        f"pearson_r={overall_summary['pearson_r']:.6f}, "
        f"spearman_rho={overall_summary['spearman_rho']:.6f}"
    )
    print("[INFO] Per-step summary:")
    print(
        summary_by_step[
            ["step", "n_samples", "pearson_r", "spearman_rho", "euclidean_mean", "dtw_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
