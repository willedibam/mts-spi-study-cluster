"""Run the frozen raw-only direct phase-coupling feasibility gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from src.oscillatory_coorganization import observed_controls
from src.oscillatory_mechanism import draw_parameters, group_contrast, phase_matrix, simulate
from src.representation_state_data import file_hash, observed_view


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main(config, output):
    if output.exists():
        raise FileExistsError(output)
    cfg = yaml.safe_load(config.read_text())
    settings, sampling, gate = cfg["generator"], cfg["sampling"], cfg["gates"]
    output.mkdir(parents=True)
    records, convergence, uncoupled, finite_arrays, reference_matrices = [], [], [], [], []
    for draw in range(sampling["paired_nuisance_draws"]):
        parameter_seed = [sampling["seed"], draw, 0]
        reference_seed = [sampling["seed"], draw, 1]
        finite_seed = [sampling["seed"], draw, 2]
        parameters = draw_parameters(parameter_seed, settings)
        for aligned in [False, True]:
            reference, latent = simulate(aligned, reference_seed, parameters, settings,
                                         t=sampling["reference_T"], return_latent=True)
            pm = phase_matrix(latent["phase"])
            am = np.corrcoef(latent["logamp"].T)
            observed = observed_controls(reference)
            reference_matrices.append([pm, am, observed["phase_matrix"], observed["envelope_matrix"]])
            half_controls = [observed_controls(half) for half in np.array_split(reference, 2)]
            record = dict(draw=draw, aligned=int(aligned), parameters=parameters,
                          parameter_seed=parameter_seed, reference_seed=reference_seed,
                          finite_seed=finite_seed, reference_sha256=array_hash(reference),
                          reference_half_agreement=[float(c["direct_agreement"][0]) for c in half_controls],
                          latent_phase_contrast=group_contrast(pm, latent["phase_group"]),
                          latent_envelope_contrast=group_contrast(am, latent["envelope_group"]),
                          observed_phase_contrast=group_contrast(observed["phase_matrix"], latent["phase_group"]),
                          observed_envelope_contrast=group_contrast(observed["envelope_matrix"], latent["envelope_group"]),
                          latent_half_phase_contrast=[group_contrast(phase_matrix(half), latent["phase_group"])
                                                     for half in np.array_split(latent["phase"], 2)])
            finite, finite_latent = simulate(aligned, finite_seed, parameters, settings,
                                             t=sampling["finite_T"], return_latent=True)
            finite_arrays.append(finite)
            record["finite_sha256"] = array_hash(finite)
            record["finite_agreement"] = {
                str(m): float(observed_controls(observed_view(finite, m, t))["direct_agreement"][0])
                for m, t in [(16, 1000), (8, 500)]
            }
            records.append(record)
            if draw < sampling["convergence_draws"]:
                fine, fine_latent = simulate(aligned, finite_seed, parameters, settings,
                                             t=sampling["finite_T"], dt=settings["fine_dt"], return_latent=True)
                offdiag = ~np.eye(32, dtype=bool)
                delta = (phase_matrix(finite_latent["phase"]) - phase_matrix(fine_latent["phase"]))[offdiag]
                convergence.append(dict(draw=draw, aligned=int(aligned), phase_edge_RMSE=float(np.sqrt(np.mean(delta**2))),
                    agreement_difference={str(m): abs(record["finite_agreement"][str(m)] - float(
                        observed_controls(observed_view(fine, m, t))["direct_agreement"][0]))
                        for m, t in [(16, 1000), (8, 500)]}))
        if draw < sampling["uncoupled_draws"]:
            _, latent_off = simulate(False, reference_seed, parameters, settings,
                                     t=sampling["reference_T"], coupling=0., return_latent=True)
            uncoupled.append(dict(draw=draw, phase_contrast=group_contrast(
                phase_matrix(latent_off["phase"]), latent_off["phase_group"])))
        print(f"Completed nuisance draw {draw + 1}/{sampling['paired_nuisance_draws']}", flush=True)

    truth = np.array([r["aligned"] for r in records])
    reference_scores = np.array([r["reference_half_agreement"] for r in records])
    reference_auc = [float(roc_auc_score(truth, reference_scores[:, half])) for half in range(2)]
    class_fractions = {
        "aligned": np.mean(reference_scores[truth == 1] > gate["aligned_minimum_agreement"], axis=0).tolist(),
        "crossed": np.mean(reference_scores[truth == 0] < gate["crossed_maximum_agreement"], axis=0).tolist(),
    }
    finite_auc = {str(m): float(roc_auc_score(truth, [r["finite_agreement"][str(m)] for r in records]))
                  for m in [16, 8]}
    gates = dict(
        reference_AUROC=min(reference_auc) >= gate["reference_minimum_AUROC"],
        reference_class_intervals=min(class_fractions["aligned"] + class_fractions["crossed"]) >= gate["minimum_class_pass_fraction"],
        latent_phase_contrast=min(r["latent_phase_contrast"] for r in records) >= gate["minimum_phase_within_between"],
        latent_envelope_contrast=min(r["latent_envelope_contrast"] for r in records) >= gate["minimum_envelope_within_between"],
        finite_AUROC=min(finite_auc.values()) >= gate["finite_minimum_AUROC"],
        integration_agreement=max(v for r in convergence for v in r["agreement_difference"].values()) <= gate["maximum_step_agreement_difference"],
        integration_phase=max(r["phase_edge_RMSE"] for r in convergence) <= gate["maximum_step_phase_edge_RMSE"],
        coupling_removal=max(abs(r["phase_contrast"]) for r in uncoupled) <= gate["maximum_uncoupled_absolute_phase_contrast"],
    )
    np.savez_compressed(output / "raw-diagnostics.npz", finite=np.array(finite_arrays),
                        reference_matrices=np.array(reference_matrices), labels=truth)
    report = dict(passed=all(gates.values()), gates=gates, reference_AUROC=reference_auc,
                  reference_class_pass_fractions=class_fractions, finite_AUROC=finite_auc,
                  config_sha256=file_hash(config), code_sha256={p: file_hash(Path(p)) for p in [
                      "scripts/scout_oscillatory_mechanism.py", "src/oscillatory_mechanism.py",
                      "src/oscillatory_coorganization.py", "src/representation_state_data.py"]},
                  artifacts={"raw-diagnostics.npz": file_hash(output / "raw-diagnostics.npz")},
                  records=records, convergence=convergence, uncoupled=uncoupled)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["passed", "gates", "reference_AUROC", "reference_class_pass_fractions", "finite_AUROC"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.output)
