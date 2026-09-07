"""Raw model and memory controls before any SPI computation for interaction share."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from scripts.check_interaction_share_feasibility import drift_and_jacobian, interaction_share, ring
from src.interaction_share_reference import energy_share, fit_map, own_memory


def parameters(value, parameterization, seed):
    if parameterization == "fixed_sum":
        return value, .8 - value, .8
    if parameterization != "independent_gain":
        raise ValueError(parameterization)
    # Independent contraction nuisance; linear q is value, not a function of a alone.
    gain = np.random.default_rng(seed).uniform(.25, .8)
    ratio = np.sqrt(2 * value) / (np.sqrt(1 - value) + np.sqrt(2 * value))
    return gain * (1 - ratio), gain * ratio, gain


def simulate(family, a, b, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    x, w = np.zeros(32), ring(32)
    states, jac = [], []
    for t in range(2300):
        drift, j = drift_and_jacobian(x, a, b, w, family)
        if t >= 300:
            states.append(x.copy()); jac.append(j)
        x = drift + .5 * rng.normal(size=32)
    return np.asarray(states), np.asarray(jac)


def run(output, parameterization):
    if output.exists():
        raise FileExistsError(output)
    rows, masters = [], []
    seed = 260908 if parameterization == "fixed_sum" else 260909
    for f, family in enumerate(["linear", "tanh"]):
        for k, value in enumerate([.1, .25, .4, .55, .7] if parameterization == "fixed_sum" else [.1, .3, .5, .7, .9]):
            for replicate in range(8):
                a, b, gain = parameters(value, parameterization, [seed, f, k, replicate, 2])
                states, jac = simulate(family, a, b, [seed, f, k, replicate])
                target = interaction_share(jac[1000:])
                master_id = f"{family}-k{k}-r{replicate}"
                sensor_order = np.random.default_rng([314159, f, k, replicate]).permutation(32)
                masters.append({"master_id": master_id, "family": family, "setting": k, "replicate": replicate,
                                "a": a, "b": b, "gain": gain, "target": target})
                for m, t in [(32, 1000), (16, 1000), (8, 500)]:
                    indices = sensor_order[:m]
                    raw = np.ascontiguousarray(states[1000-t:1000, indices])
                    future = states[1000:, indices]
                    truth_j = jac[1000-t:1000][:, indices][:, :, indices]
                    own = np.square(np.diagonal(truth_j, axis1=1, axis2=2)).sum(axis=1).mean()
                    cross = np.square(truth_j).sum(axis=(1, 2)).mean() - own
                    row = {**masters[-1], "M": m, "T": t, "sensor_indices": indices.tolist(),
                           "own_memory": own_memory(raw),
                           "oracle_restricted_share": energy_share(own, cross, m),
                           "oracle_sampling_adjusted": energy_share(own, cross, m, 32)}
                    for nonlinear in [False, True]:
                        name = "nonlinear" if nonlinear else "linear"
                        model = fit_map(raw, nonlinear=nonlinear)
                        d, o = model.energies(raw)
                        row[name] = energy_share(d, o, m)
                        row[name + "_sampling_adjusted"] = energy_share(d, o, m, 32)
                        row[name + "_future_MSE"] = float(np.mean((model.predict(future[:-1]) - future[1:])**2))
                    rows.append(row)
    summary = {}
    methods = ["linear", "linear_sampling_adjusted", "nonlinear", "nonlinear_sampling_adjusted",
               "oracle_restricted_share", "oracle_sampling_adjusted"]
    for family in ["linear", "tanh"]:
        for m, t in [(32, 1000), (16, 1000), (8, 500)]:
            subset = [r for r in rows if r["family"] == family and (r["M"], r["T"]) == (m, t)]
            target = np.asarray([r["target"] for r in subset])
            summary[f"{family}/M{m}-T{t}"] = {method: {
                "MAE": float(np.mean(abs(np.asarray([r[method] for r in subset]) - target))),
                "bias": float(np.mean(np.asarray([r[method] for r in subset]) - target)),
                "spearman": float(spearmanr([r[method] for r in subset], target).statistic)} for method in methods}
    # One fixed 20-label diagnostic per source family. Evaluation masters never
    # train a calibrator in that direction, including their other observation views.
    calibrated = []
    for family in ["linear", "tanh"]:
        training = [r for r in rows if r["family"] == family and r["replicate"] < 4 and (r["M"], r["T"]) == (16, 1000)]
        evaluation = [r for r in rows if r["replicate"] >= 4]
        assert len(training) == 20
        assert not {r["master_id"] for r in training} & {r["master_id"] for r in evaluation}
        median = float(np.median([r["target"] for r in training]))
        for row in evaluation:
            calibrated.append({"source_family": family, "method": "source_median", "master_id": row["master_id"],
                               "family": row["family"], "M": row["M"], "T": row["T"],
                               "target": row["target"], "prediction": median})
        for method in ["own_memory", "linear_sampling_adjusted", "nonlinear_sampling_adjusted"]:
            model = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
                [r[method] for r in training], [r["target"] for r in training])
            for row, prediction in zip(evaluation, model.predict([r[method] for r in evaluation])):
                calibrated.append({"source_family": family, "method": method, "master_id": row["master_id"],
                                   "family": row["family"], "M": row["M"], "T": row["T"],
                                   "target": row["target"], "prediction": float(prediction)})
    calibration_summary = {}
    for source in ["linear", "tanh"]:
        for destination in ["linear", "tanh"]:
            for m, t in [(32, 1000), (16, 1000), (8, 500)]:
                subset = [r for r in calibrated if r["source_family"] == source and r["family"] == destination and (r["M"], r["T"]) == (m, t)]
                calibration_summary[f"{source}->{destination}/M{m}-T{t}"] = {
                    method: float(np.mean([abs(r["prediction"] - r["target"]) for r in subset if r["method"] == method]))
                    for method in ["source_median", "own_memory", "linear_sampling_adjusted", "nonlinear_sampling_adjusted"]}
    result = {"status": "exploratory_raw_reference_audit_no_SPI", "parameterization": parameterization,
              "master_seed": seed, "ridge_fraction": .001, "population": 32,
              "source_calibration_labels": 20, "calibration_split": "replicates0:3 source; evaluation4:7",
              "summary": summary, "calibration_summary": calibration_summary, "rows": rows,
              "calibrated_predictions": calibrated, "masters": masters,
              "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
                  __file__, "scripts/check_interaction_share_feasibility.py", "src/interaction_share_reference.py"]}}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Raw reference audit", "", f"Parameterization: {parameterization}. 80 masters; 40/family; no SPI evaluation.", "",
             "Uncalibrated input-derived estimates; MAE against future full-population q. Oracle rows are diagnostics, not predictors.", "",
             "| Family / observation | Linear | Linear adjusted | Nonlinear | Nonlinear adjusted | Oracle restricted | Oracle adjusted |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cell, values in summary.items():
        lines.append(f"| {cell} | " + " | ".join(f"{values[m]['MAE']:.4f}" for m in methods) + " |")
    lines += ["", "Single-feature monotone calibrators fitted on 20 labelled source masters (M16/T1000), with disjoint evaluation masters:", "",
              "| Transfer / observation | Source median | Own-channel memory | Linear adjusted | Nonlinear adjusted |", "|---|---:|---:|---:|---:|"]
    for cell, values in calibration_summary.items():
        lines.append(f"| {cell} | " + " | ".join(f"{v:.4f}" for v in values.values()) + " |")
    lines += ["", "Adjusted estimates correct different dyad/node sampling fractions. Neither their ratio nor fitted "
              "coefficients are guaranteed unbiased. Hidden-variable effects and finite-sample squared-coefficient bias "
              "remain. Calibration is an exploratory diagnostic, not a frozen learning-curve comparison. "
              "Full-population input is a reference condition; the planned source remains M16/T1000.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(calibration_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameterization", choices=["fixed_sum", "independent_gain"], default="fixed_sum")
    args = parser.parse_args()
    run(args.output, args.parameterization)
