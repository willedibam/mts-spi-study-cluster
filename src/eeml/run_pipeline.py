"""
EEML pipeline: SPI data → graph construction → train MPNN → evaluate.

Usage:
    python -m src.eeml.run_pipeline \\
        --data-dir data/eeml/chat/ \\
        --generators chat-a \\
        --seeds 5 \\
        --device cpu

Stages:
    1. Load computed SPI data from run_experiments output
    2. Filter SPI dimensions, fit robust scaler on training data
    3. Build PyG graphs
    4. Train main model + baselines across seeds
    5. Save results and run log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from ..utils import load_json, project_root, to_relative
from .features import node_features
from .graph_build import (
    SPIScaler,
    build_graph,
    filter_spi_dimensions,
    load_spi_tensor,
)
from .model import EdgeAblationMPNN, SPIEdgeMPNN
from .train import TrainConfig, TrainResult, split_data, train_model


# ---------------------------------------------------------------------------
# Generator family groupings
# ---------------------------------------------------------------------------

_GENERATOR_GROUPS: dict[str, list[str]] = {
    "chat-a": ["chat-a-chain", "chat-a-fork", "chat-a-collider"],
    "chat-b": ["chat-b-no-direct", "chat-b-with-direct"],
    "chat-c": ["chat-c-linear", "chat-c-tanh"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEML GNN pipeline")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory (e.g. data/eeml/chat/)",
    )
    p.add_argument(
        "--generators",
        nargs="+",
        default=["chat-a"],
        help="Generator groups to train on (chat-a, chat-b, chat-c, or all)",
    )
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    p.add_argument("--device", default="cpu", help="torch device")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--top-d", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline models (only run main SPI-MPNN)",
    )
    p.add_argument(
        "--output-dir",
        help="Output directory for results. Default: <data-dir>/eeml_results/",
    )
    return p.parse_args(argv)


def _discover_datasets(
    data_dir: Path, class_names: list[str]
) -> dict[str, list[Path]]:
    """
    Find dataset directories per class under data_dir/<class_name>/.

    Returns {class_name: [dataset_dir, ...]} sorted by directory name.
    """
    result: dict[str, list[Path]] = {}
    for name in class_names:
        class_dir = data_dir / name
        if not class_dir.exists():
            print(f"[WARN] Class directory not found: {class_dir}")
            continue
        dirs = sorted(
            d
            for d in class_dir.iterdir()
            if d.is_dir() and (d / "spi_mpis.npz").exists() and (d / "meta.json").exists()
        )
        if dirs:
            result[name] = dirs
        else:
            print(f"[WARN] No completed datasets in {class_dir}")
    return result


def _load_spi_names(dataset_dir: Path) -> list[str]:
    """Extract ordered SPI names from meta.json."""
    meta = load_json(dataset_dir / "meta.json")
    spis = meta.get("pyspi", {}).get("spis", [])
    return [s["name"] for s in spis if isinstance(s, dict) and "name" in s]


def _load_all_data(
    datasets_by_class: dict[str, list[Path]],
    spi_names: list[str],
    class_to_label: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[Path]]:
    """
    Load all SPI tensors, MTS arrays, and labels.

    Returns:
        spi_tensors: list of (M, M, K) arrays
        mts_arrays: list of (T, M) arrays
        labels: list of int labels
        paths: list of dataset paths
    """
    spi_tensors = []
    mts_arrays = []
    labels = []
    paths = []

    for class_name, dirs in datasets_by_class.items():
        label = class_to_label[class_name]
        for d in dirs:
            try:
                tensor = load_spi_tensor(d, spi_names)
                mts = np.load(d / "timeseries.npy").astype(np.float64)
                spi_tensors.append(tensor)
                mts_arrays.append(mts)
                labels.append(label)
                paths.append(d)
            except Exception as e:
                print(f"[WARN] Skipping {d}: {e}")

    return spi_tensors, mts_arrays, labels, paths


def _build_dataset(
    spi_tensors: list[np.ndarray],
    mts_arrays: list[np.ndarray],
    labels: list[int],
) -> list[Data]:
    """Convert arrays to PyG Data objects."""
    dataset = []
    for tensor, mts, label in zip(spi_tensors, mts_arrays, labels):
        data = build_graph(tensor, mts, label)
        dataset.append(data)
    return dataset


def _run_experiment(
    generator_group: str,
    data_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Run full pipeline for one generator group."""
    class_names = _GENERATOR_GROUPS[generator_group]
    print(f"\n{'='*70}")
    print(f"Generator group: {generator_group}")
    print(f"Classes: {class_names}")
    print(f"{'='*70}")

    # Discover datasets
    datasets_by_class = _discover_datasets(data_dir, class_names)
    if not datasets_by_class:
        print(f"[ERROR] No data found for {generator_group}")
        return {}

    # Class label mapping
    class_to_label = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    for name, dirs in datasets_by_class.items():
        print(f"  {name}: {len(dirs)} datasets")

    # Get SPI names from first dataset
    first_dir = next(iter(datasets_by_class.values()))[0]
    all_spi_names = _load_spi_names(first_dir)
    print(f"  SPI dimensions: {len(all_spi_names)}")

    # Load all data
    print("[STAGE] Loading SPI tensors and MTS data...")
    spi_tensors, mts_arrays, labels, paths = _load_all_data(
        datasets_by_class, all_spi_names, class_to_label
    )
    print(f"  Loaded {len(spi_tensors)} samples")

    if len(spi_tensors) == 0:
        print("[ERROR] No valid samples loaded")
        return {}

    # Split indices
    n = len(spi_tensors)
    rng = np.random.default_rng(0)
    indices = rng.permutation(n)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_tensors = [spi_tensors[i] for i in train_idx]

    # Filter SPI dimensions on training data
    print("[STAGE] Filtering SPI dimensions...")
    retained_names, retained_indices = filter_spi_dimensions(
        all_spi_names, train_tensors
    )

    if not retained_names:
        print("[ERROR] All SPI dimensions were dropped")
        return {}

    # Subset tensors to retained dimensions
    spi_tensors = [t[:, :, retained_indices] for t in spi_tensors]
    K = len(retained_names)

    # Fit scaler on training data
    print("[STAGE] Fitting robust scaler...")
    scaler = SPIScaler()
    train_tensors_filtered = [spi_tensors[i] for i in train_idx]
    scaler.fit(train_tensors_filtered)

    # Scale all data
    spi_tensors = [scaler.transform(t) for t in spi_tensors]

    # Build PyG dataset
    print("[STAGE] Building PyG graphs...")
    dataset = _build_dataset(spi_tensors, mts_arrays, labels)

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    n_node_features = train_data[0].x.shape[1]

    # Training config
    config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
    )

    # Run across seeds
    all_results: dict[str, list[TrainResult]] = defaultdict(list)
    models_to_run = ["spi-mpnn"]
    if not args.skip_baselines:
        models_to_run.extend(["edge-ablation"])

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for model_name in models_to_run:
            print(f"\n  Model: {model_name}")

            if model_name == "spi-mpnn":
                model = SPIEdgeMPNN(
                    n_spi=K,
                    n_node_features=n_node_features,
                    n_classes=n_classes,
                    hidden=args.hidden,
                    n_layers=args.n_layers,
                    top_d=args.top_d,
                    dropout=args.dropout,
                )
            elif model_name == "edge-ablation":
                model = EdgeAblationMPNN(
                    n_spi=K,
                    n_node_features=n_node_features,
                    n_classes=n_classes,
                    hidden=args.hidden,
                    n_layers=args.n_layers,
                    top_d=args.top_d,
                    dropout=args.dropout,
                )
            else:
                continue

            result = train_model(model, train_data, val_data, test_data, config)
            all_results[model_name].append(result)

    # Summary
    summary: dict[str, Any] = {
        "generator_group": generator_group,
        "classes": class_names,
        "n_samples": len(dataset),
        "n_spi": K,
        "spi_names": retained_names,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "models": {},
    }

    print(f"\n{'='*70}")
    print(f"Summary for {generator_group}")
    print(f"{'='*70}")
    for model_name, results in all_results.items():
        f1s = [r.test_f1 for r in results]
        accs = [r.test_acc for r in results]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        print(f"  {model_name:20s}  F1={mean_f1:.4f}±{std_f1:.4f}  Acc={mean_acc:.4f}±{std_acc:.4f}")

        model_summary: dict[str, Any] = {
            "test_f1_mean": float(mean_f1),
            "test_f1_std": float(std_f1),
            "test_acc_mean": float(mean_acc),
            "test_acc_std": float(std_acc),
            "per_seed": [],
        }
        for r in results:
            seed_info: dict[str, Any] = {
                "test_f1": r.test_f1,
                "test_acc": r.test_acc,
                "best_epoch": r.best_epoch,
                "best_val_f1": r.best_val_f1,
                "train_seconds": r.train_seconds,
            }
            if r.learned_w.size > 0:
                seed_info["learned_w"] = r.learned_w.tolist()
                seed_info["learned_b"] = r.learned_b
            model_summary["per_seed"].append(seed_info)

        summary["models"][model_name] = model_summary

    # Save results
    result_path = output_dir / f"{generator_group}_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {to_relative(result_path)}")

    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root() / data_dir

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "eeml_results"
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir

    # Resolve generator groups
    groups: list[str] = []
    for g in args.generators:
        if g == "all":
            groups.extend(_GENERATOR_GROUPS.keys())
        elif g in _GENERATOR_GROUPS:
            groups.append(g)
        else:
            print(f"[ERROR] Unknown generator group '{g}'. Available: {list(_GENERATOR_GROUPS.keys())}")
            sys.exit(1)

    print(f"Data directory: {to_relative(data_dir)}")
    print(f"Output directory: {to_relative(output_dir)}")
    print(f"Generator groups: {groups}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")

    for group in groups:
        _run_experiment(group, data_dir, args, output_dir)


if __name__ == "__main__":
    main()
