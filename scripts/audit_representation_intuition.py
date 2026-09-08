"""Describe existing intuition cases; no training, simulation, or pyspi rerun.

Run from the repository root. Results are exploratory summaries of the saved
case-study records, not a representation-learning benchmark or variance proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew


CASES = {
    "filter_roll": (
        "data/r_rho_mi/260703_g-roll",
        ["cov_EmpiricalCovariance", "spearmanr", "mi_kraskov_NN-4"],
    ),
    "lag_warp": (
        "data/dtw_euclidean/260413_2_lagged-warping_rmse_T1000",
        ["pdist_euclidean_rmse", "xpdist_euclidean_tau-10_min_rmse", "dtw_rmse"],
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_correlations(values):
    """Use every ordered off-diagonal entry, preserving both directions."""
    m = values.shape[1]
    edge_values = values[:, ~np.eye(m, dtype=bool)]
    if not np.isfinite(edge_values).all():
        raise ValueError("These three-SPI cases must have finite off-diagonals")
    if (edge_values.std(axis=1) < 1e-12).any():
        raise ValueError("Constant MPI: cannot interpret its Pearson correlation")
    return np.corrcoef(edge_values)[np.triu_indices(len(values), 1)]


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    result = {
        "status": "descriptive_existing_case_audit_not_a_benchmark",
        "script_sha256": digest(Path(__file__)),
        "numpy": np.__version__,
        "cases": {},
    }
    for name, (relative_path, keys) in CASES.items():
        rows = []
        paths = sorted((root / relative_path).glob("*/*/timeseries.npy"))
        if not paths:
            raise FileNotFoundError(root / relative_path)
        for path in paths:
            meta_path, mpi_path = path.parent / "meta.json", path.parent / "spi_mpis.npz"
            meta = json.loads(meta_path.read_text())
            x = np.load(path, allow_pickle=False)
            if x.shape != (meta["T"], meta["M"]) or not np.isfinite(x).all():
                raise ValueError(f"Unexpected input: {path}")
            with np.load(mpi_path, allow_pickle=False) as bank:
                matrices = np.stack([bank[k] for k in keys])
            z = pair_correlations(matrices)
            centered = x - x.mean(axis=0)
            lag1 = np.sum(centered[:-1] * centered[1:], axis=0) / np.sqrt(
                np.sum(centered[:-1] ** 2, axis=0) * np.sum(centered[1:] ** 2, axis=0)
            )
            metrics = {
                "mean_channel_abs_skew": float(np.abs(skew(x, axis=0)).mean()),
                "mean_channel_excess_kurtosis": float(kurtosis(x, axis=0).mean()),
                "mean_channel_lag1": float(lag1.mean()),
                **{f"pearson_{keys[i]}__{keys[j]}": float(v)
                   for (i, j), v in zip(zip(*np.triu_indices(3, 1)), z)},
            }
            if name == "filter_roll":
                transformed = matrices.copy()
                transformed[2] = np.sqrt(np.clip(-np.expm1(-2 * np.maximum(matrices[2], 0)), 0, 1))
                zl = pair_correlations(transformed)
                metrics["max_abs_pearson_change_linfoot"] = float(np.max(np.abs(zl - z)))
            rows.append({
                "record": str(path.parent.relative_to(root)),
                "class": meta["mts_class"], "seed": meta["generator"]["seed"],
                "sha256": {p.name: digest(p) for p in [path, meta_path, mpi_path]},
                "metrics": metrics,
            })
        groups = {}
        for label in sorted({r["class"] for r in rows}):
            selected = [r for r in rows if r["class"] == label]
            groups[label] = {
                "n": len(selected),
                "metrics": {key: {"mean": float(np.mean([r["metrics"][key] for r in selected])),
                                  "sd_across_records": float(np.std([r["metrics"][key] for r in selected], ddof=1))}
                            for key in selected[0]["metrics"]},
            }
        result["cases"][name] = {"spis": keys, "n_records": len(rows),
                                   "unique_seeds": len({r["seed"] for r in rows}),
                                   "groups": groups, "records": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["groups"] for k, v in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("results/representation_intuition_260909/audit.json"))
    args = parser.parse_args()
    run(args.root.resolve(), args.output)
