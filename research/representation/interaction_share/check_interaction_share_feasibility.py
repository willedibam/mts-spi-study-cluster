"""Raw-only feasibility of a shared local-sensitivity target; no SPI evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def ring(n):
    return (np.roll(np.eye(n), 1, axis=1) + np.roll(np.eye(n), -1, axis=1)) / 2


def drift_and_jacobian(x, a, b, w, family):
    if family == "linear":
        values, derivative = x, np.ones_like(x)
    elif family == "tanh":
        values = np.tanh(x)
        derivative = 1 - values**2
    else:
        raise ValueError(family)
    return a * x + b * (w @ values), a * np.eye(len(x)) + b * w * derivative[None, :]


def interaction_share(jacobians):
    energy = np.square(jacobians)
    total = energy.sum(axis=(-1, -2)).mean()
    own = np.square(np.diagonal(jacobians, axis1=-2, axis2=-1)).sum(axis=-1).mean()
    return float((total - own) / total)


def check(output):
    if output.exists():
        raise FileExistsError(output)
    n, samples, burn, sigma = 32, 1000, 300, .5
    w = ring(n)
    records = []
    for family_index, family in enumerate(["linear", "tanh"]):
        for parameter_index, a in enumerate([.1, .25, .4, .55, .7]):
            b = .8 - a
            for replicate in range(8):
                rng = np.random.default_rng(np.random.SeedSequence([260908, family_index, parameter_index, replicate]))
                x = np.zeros(n)
                states, jacobians = [], []
                for t in range(burn + 2 * samples):
                    drift, jacobian = drift_and_jacobian(x, a, b, w, family)
                    if t >= burn:
                        states.append(x.copy()); jacobians.append(jacobian)
                    x = drift + sigma * rng.normal(size=n)
                states, jacobians = np.asarray(states), np.asarray(jacobians)
                past = interaction_share(jacobians[:samples])
                future = interaction_share(jacobians[samples:])
                records.append({"family": family, "a": a, "b": b, "replicate": replicate,
                                "past_share": past, "future_share": future,
                                "input_std": float(states[:samples].std()),
                                "max_abs_state": float(abs(states).max()),
                                "mean_squared_nonlinearity_derivative": float(np.mean((1 - np.tanh(states[:samples])**2)**2))
                                if family == "tanh" else 1.0})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "raw_only_target_feasibility_not_a_benchmark_result", "N_full": n,
              "input_samples": samples, "future_label_samples": samples, "burn": burn, "noise_std": sigma,
              "lipschitz_upper_bound": .8, "independent_realizations": len(records), "master_seed": 260908,
              "numpy": np.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "target": "future mean squared off-diagonal drift-Jacobian energy / future mean squared full drift-Jacobian energy",
              "limitation": "This is coordinate-dependent local sensitivity in fixed state units and discrete time, not a universal coupling order parameter.",
              "records": records}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({family: {"target_range": [min(r["future_share"] for r in records if r["family"] == family),
                                               max(r["future_share"] for r in records if r["family"] == family)],
                              "past_future_MAE": float(np.mean([abs(r["past_share"] - r["future_share"]) for r in records if r["family"] == family]))}
                      for family in ["linear", "tanh"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output)
