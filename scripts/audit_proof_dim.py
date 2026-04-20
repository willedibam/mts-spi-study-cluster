"""Audit n_pcs_95 across every class in proof.yaml.

Informational: which classes pass Cliff's filter (n_pcs_95 >= 3) on
which (M, T) cells, what the median dim is, and whether intrinsic
low-dim is a feature of the regime or an artefact of params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dimensionality import n_pcs_for_variance
from src.mapping import DatasetMapping, ExperimentConfig
from src.run_experiments import generate_synthetic_from_spec

ALPHA = 0.95
N_PCS_MIN = 3


def main():
    cfg = ExperimentConfig.from_file(ROOT / "configs/generate/embeddings/proof.yaml")
    cfg.timestamp = False
    mapping = DatasetMapping(cfg)
    print(f"Total specs: {len(mapping.specs)}")

    rows = []
    for i, spec in enumerate(mapping.specs):
        try:
            data, _ = generate_synthetic_from_spec(spec)
            npcs = n_pcs_for_variance(data, alpha=ALPHA)
        except Exception as e:
            npcs = -1
            print(f"  [warn] {spec.mts_class} M={spec.M} T={spec.T} I={spec.instance}: {e}")
        rows.append({
            "class": spec.mts_class,
            "M": spec.M,
            "T": spec.T,
            "I": spec.instance,
            "n_pcs_95": npcs,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(mapping.specs)}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "scripts" / "audit_proof_dim_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n")

    print("=" * 80)
    print("Median n_pcs_95 per (class, M, T)")
    print("=" * 80)
    pivot_med = df.pivot_table(
        index="class", columns=["M", "T"], values="n_pcs_95", aggfunc="median"
    )
    print(pivot_med.to_string())

    print("\n" + "=" * 80)
    print(f"Pass-rate (n_pcs_95 >= {N_PCS_MIN}) per (class, M, T)")
    print("=" * 80)
    pivot_pass = df.pivot_table(
        index="class",
        columns=["M", "T"],
        values="n_pcs_95",
        aggfunc=lambda x: (x >= N_PCS_MIN).mean(),
    )
    print(pivot_pass.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n" + "=" * 80)
    print(f"Class-level summary (across all M, T, I)")
    print("=" * 80)
    summary = df.groupby("class").agg(
        n_datasets=("n_pcs_95", "size"),
        median_npcs=("n_pcs_95", "median"),
        min_npcs=("n_pcs_95", "min"),
        max_npcs=("n_pcs_95", "max"),
        pass_rate=("n_pcs_95", lambda x: (x >= N_PCS_MIN).mean()),
    )
    summary = summary.sort_values("median_npcs")
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n" + "=" * 80)
    print("Classes failing Cliff's filter (pass_rate < 1.0)")
    print("=" * 80)
    failing = summary[summary["pass_rate"] < 1.0]
    if len(failing):
        print(failing.to_string(float_format=lambda v: f"{v:.2f}"))
    else:
        print("(none — all classes pass on all (M, T, I) cells)")

    print("\n" + "=" * 80)
    print("Effective dim ratio (median_npcs / M) — flags classes whose intrinsic")
    print("dim doesn't grow with M (i.e. low-dim attractor regardless of channel count)")
    print("=" * 80)
    pivot_ratio = df.copy()
    pivot_ratio["ratio"] = pivot_ratio["n_pcs_95"] / pivot_ratio["M"]
    ratio_table = pivot_ratio.pivot_table(
        index="class", columns="M", values="ratio", aggfunc="median"
    )
    print(ratio_table.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
