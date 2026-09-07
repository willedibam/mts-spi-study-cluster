"""Cheap raw-only validation of the proposed fixed-population observation pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.generators.order_parameter import generate_kuramoto_order_parameter, kuramoto_critical_coupling


def check(config_path: Path, output: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    params = config["generator"].copy()
    params.pop("name")
    kappas = params.pop("reduced_couplings")
    params.pop("coupling_scale")
    records = []
    # Dedicated diagnostic seed namespace, disjoint from proposed fit/eval data.
    for index in (0, 2, 4, 7):
        for replicate in (0, 1):
            for burn in (100., 200.):
                seed = np.random.SeedSequence([config["sampling"]["master_seed"], 9, index, replicate])
                options = {**params, "burn_time": burn,
                           "K": kappas[index] * kuramoto_critical_coupling(params["frequency_distribution"], params["omega_std"])}
                observed, internals = generate_kuramoto_order_parameter(
                    **options, rng=np.random.default_rng(seed), return_internals=True)
                views = {(m, t): observed[-t:, :m] for m in config["observations"]["M_values"] for t in config["observations"]["T_values"]}
                assert observed.shape == (1000, 32) and np.isfinite(observed).all()
                for (m, t), view in views.items():
                    assert view.shape == (t, m)
                    np.testing.assert_array_equal(view, views[(32, 1000)][-t:, :m])
                assert len(internals.r_full_future) == 1000
                target = float(internals.r_full_future.mean())
                assert 0 <= target <= 1
                records.append({"coupling_index": index, "kappa": kappas[index], "replicate": replicate,
                                "burn_time": burn, "future_coherence": target,
                                "past_full_coherence": float(internals.r_full.mean()),
                                "minimum_channel_std": float(observed.std(axis=0).min()),
                                "views": len(views), "view_future_target_is_shared": True})
                print(f"kappa={kappas[index]} replicate={replicate} burn={burn:g} views OK", flush=True)
    report = {"status": "raw_view_construction_checked_no_pyspi_or_model_training",
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "generator_sha256": hashlib.sha256(Path("src/generators/order_parameter.py").read_bytes()).hexdigest(),
              "records": records,
              "target_range": [min(r["future_coherence"] for r in records), max(r["future_coherence"] for r in records)],
              "warning": "This checks construction and records burn-time sensitivity; it does not establish stationarity, predictive utility, or superiority."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.config, args.output)
