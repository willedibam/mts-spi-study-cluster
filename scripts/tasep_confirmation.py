"""Sealed N=32 open-TASEP fresh-seed confirmation of a frozen SPI coordinate.

The physical generator, control grid and gates are unchanged from the pilot.
Each case is one NPZ with metadata, binary observations and reference diagnostics.
No fit, exclusions, numerical noise or thermodynamic finite-N jump is introduced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts import tasep_phase_boundary as generator

PROTOCOL = dict(beta=.2, burn=100000., observation_steps=2000, dt=1.,
                reference_time=1000000., reference_blocks=32, reference_trace_dt=10., initial="random")
MODEL_FILES = ("model.npz", "geometry.json", "summary.json")
MODEL_ARRAYS = ("keep", "impute", "center", "component", "score_scale", "spi_order")
P90_CONFIG = Path(__file__).resolve().parents[1] / "configs/pyspi/benchmarked_p90.yaml"
SOURCE = "https://www.lps.ens.fr/~derrida/PAPIERS/1993/DEHP-93.pdf"
GATES = dict(raw_failure_policy="stop without exclusions", minimum_endpoint_contrast=.3,
             minimum_control_mean_spearman=.9, maximum_exact_error_p95=.05,
             maximum_half_difference_p95=.08)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cases():
    alphas = list(dict.fromkeys(c["alpha"] for c in generator.cases_from_config({}) if c["N"] == 32))
    return [dict(N=32, alpha=alpha, seed=seed, role="evaluation")
            for alpha in alphas for seed in range(2609152001, 2609152033)]


def model_identity(frozen):
    files = {name: digest(frozen / name) for name in MODEL_FILES}
    arrays = {}
    with np.load(frozen / "model.npz", allow_pickle=False) as model:
        require(set(model.files) == set(MODEL_ARRAYS), "unexpected frozen model schema")
        for name in MODEL_ARRAYS:
            array = model[name]
            arrays[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                               sha256=hashlib.sha256(array.tobytes()).hexdigest())
    summary = json.loads((frozen / "summary.json").read_text())
    geometry = json.loads((frozen / "geometry.json").read_text())
    require(summary["passes"] and geometry["passes"], "pilot model did not pass its original gates")
    require(summary["display_sign"] in (-1, 1), "invalid frozen display orientation")
    return dict(files=files, arrays=arrays, display_sign=summary["display_sign"])


def create_seal(frozen, output, pilot_physics_gate):
    if output.exists():
        raise FileExistsError(output)
    pilot = json.loads(pilot_physics_gate.read_text())
    require(pilot["system"] == "tasep" and pilot["arms"]["32"]["passes"], "N32 pilot physics gate failed/missing")
    require(pilot["source_sha256"] == digest(generator.__file__), "generator differs from validated pilot")
    plan = dict(schema="tasep32-fresh-seed-confirmation-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases(), protocol=dict(PROTOCOL), primary_M=32, primary_N=32, primary_T=1000,
        roles="all evaluation; no development rows or refitting",
        frozen_path=str(frozen.resolve()), frozen_identity=model_identity(frozen),
        source_sha256=digest(generator.__file__), confirmation_source_sha256=digest(__file__),
        pyspi_config_sha256=digest(P90_CONFIG),
        pilot_physics_gate=dict(path=str(pilot_physics_gate.resolve()), sha256=digest(pilot_physics_gate),
                               N32=pilot["arms"]["32"]),
        physical_gates=dict(GATES),
        spi_gates="unchanged shared frozen analysis: <=5% selected-feature missingness per row; <=10% excluded overall and >=75% eligible per control cell",
        Q_definition="event-holding-time integral of full-system occupation density over disjoint future reference; finite-N exact stationary density checked independently",
        inference="fresh-seed confirmation on the same control grid; finite-N density crossover, not a thermodynamic discontinuity or new-control generalization")
    plan["seal_identity"] = canonical_hash(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(seal=str(output), seal_identity=plan["seal_identity"], cases=len(plan["cases"]))))
    return plan


def read_seal(path, frozen=None):
    plan = json.loads(path.read_text())
    contents = {key: value for key, value in plan.items() if key != "seal_identity"}
    require(plan["seal_identity"] == canonical_hash(contents), "seal contents changed")
    require(plan["cases"] == cases() and plan["protocol"] == PROTOCOL and plan["physical_gates"] == GATES,
            "confirmation design or gates changed")
    require(plan["source_sha256"] == digest(generator.__file__), "generator changed after sealing")
    require(plan["confirmation_source_sha256"] == digest(__file__), "confirmation code changed after sealing")
    require(plan["pyspi_config_sha256"] == digest(P90_CONFIG), "p90 catalogue changed after sealing")
    if frozen is not None:
        require(model_identity(frozen) == plan["frozen_identity"], "frozen model/geometry/summary/sign changed")
    return plan


def run_physics(index, root, seal):
    plan = read_seal(seal)
    require(0 <= index < len(plan["cases"]), "case index outside sealed design")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case-{index:04d}.npz"
    if path.exists():
        raise FileExistsError(path)
    case = plan["cases"][index]
    simulation_case = {key: case[key] for key in ("N", "alpha", "seed")}
    arrays, metadata = generator.simulate(simulation_case, plan["protocol"])
    metadata.update(case_index=index, role="evaluation", seal_identity=plan["seal_identity"],
        seal_file_sha256=digest(seal), frozen_model_sha256=plan["frozen_identity"]["files"]["model.npz"],
        input_sha256=hashlib.sha256(arrays["observed"].tobytes()).hexdigest())
    # Block densities are event-weighted integrals, NOT means of reference_density.
    arrays["reference_block_density"] = np.asarray(metadata["Q_blocks"], dtype=np.float64)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    print(json.dumps(dict(case_index=index, path=str(path), Q=metadata["Q_reference"],
                          elapsed_seconds=metadata["elapsed_seconds"])), flush=True)
    return path


def validate_physical_summaries(metadata, block_density, exact):
    blocks = np.asarray(block_density)
    require(blocks.shape == (32,) and np.isfinite(blocks).all() and np.all((blocks >= 0) & (blocks <= 1)),
            "invalid reference block integrals")
    require(np.array_equal(blocks, np.asarray(metadata["Q_blocks"])), "retained reference blocks differ from metadata")
    values = dict(Q_reference=float(blocks.mean()), Q_exact=exact["density"], exact_current=exact["current"],
                  Q_first_half=float(blocks[:16].mean()), Q_second_half=float(blocks[16:].mean()),
                  block_mean_se=float(blocks.std(ddof=1)/np.sqrt(32)),
                  reference_absolute_error=float(abs(blocks.mean()-exact["density"])))
    for key, value in values.items():
        require(np.isclose(metadata[key], value, atol=1e-12, rtol=0), f"physical {key} inconsistent with blocks/exact solution")
    return values


def evaluate_physics(records, limits=GATES):
    frame = pd.DataFrame(records)
    means = frame.groupby("control").Q.mean()
    require(len(means) >= 2, "missing control endpoints")
    contrast = float(means.iloc[-1] - means.iloc[0])
    rho = float(spearmanr(means.index, means.values).statistic)
    checks = dict(raw_channels_vary=bool(frame.raw_ok.all()),
        physical_contrast=bool(contrast > limits["minimum_endpoint_contrast"]),
        increasing_curve=bool(rho > limits["minimum_control_mean_spearman"]),
        exact_agreement=bool(frame.reference_error.quantile(.95) < limits["maximum_exact_error_p95"]),
        reference_stability=bool(frame.half_difference.quantile(.95) < limits["maximum_half_difference_p95"]))
    return dict(system="tasep", N=32, passes=all(checks.values()), checks=checks,
        records=len(frame), excluded_rows=0, contrast=contrast, control_mean_spearman=rho,
        reference_error_p95=float(frame.reference_error.quantile(.95)),
        half_difference_p95=float(frame.half_difference.quantile(.95)), control_mean_Q=means.to_dict())


def gate_export(physics, frozen, output, seal):
    from scripts.finite_regime_pipeline import export_arrays
    if output.exists():
        raise FileExistsError(output)
    plan = read_seal(seal, frozen)
    expected = [physics / f"case-{i:04d}.npz" for i in range(len(plan["cases"]))]
    require(set(physics.glob("case-*.npz")) == set(expected), "confirmation bank incomplete or contains extra cases")
    exact_values = {case["alpha"]: generator.exact_stationary(32, case["alpha"], .2)
                    for case in plan["cases"][::32]}
    records, rows, arrays, identities = [], [], {}, {}
    seal_file_hash = digest(seal)
    for index, (path, case) in enumerate(zip(expected, plan["cases"])):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            observed = archive["observed"]
            require(metadata["case_index"] == index, "case index mismatch")
            for key, value in {**PROTOCOL, **case}.items():
                require(metadata[key] == value, f"case {index} identity/protocol mismatch: {key}")
            require(metadata["M"] == metadata["N"] == 32 and observed.shape == (32, 2000), "full-state dimensions changed")
            require(metadata["reference_start"] == 102000., "reference no longer disjoint after maximal observation")
            require(metadata["source_sha256"] == plan["source_sha256"], "case generator source differs")
            require(metadata["seal_identity"] == plan["seal_identity"] and metadata["seal_file_sha256"] == seal_file_hash,
                    "case seal differs")
            require(metadata["frozen_model_sha256"] == plan["frozen_identity"]["files"]["model.npz"], "case model seal differs")
            require(hashlib.sha256(observed.tobytes()).hexdigest() == metadata["input_sha256"], "observation checksum differs")
            values = validate_physical_summaries(metadata, archive["reference_block_density"], exact_values[case["alpha"]])
            require(np.isclose(metadata["Q_window"], observed.mean(), atol=1e-12, rtol=0), "maximal-window mean inconsistent")
            raw_ok = bool(np.isin(observed, [0, 1]).all() and (observed[:, :1000].var(axis=1) > 0).all())
            records.append(dict(case_index=index, control=case["alpha"], seed=case["seed"], Q=values["Q_reference"],
                Q_exact=values["Q_exact"], raw_ok=raw_ok, reference_error=values["reference_absolute_error"],
                half_difference=abs(values["Q_first_half"] - values["Q_second_half"]),
                tau_int=metadata["estimated_tau_int"], effective_samples=metadata["estimated_effective_samples"]))
            short = np.ascontiguousarray(observed[:, :1000], dtype=np.float64)
        identities[path.name] = digest(path)
        row_id = f"tasep-n32-a{case['alpha']:.6f}-s{case['seed']}-t1000"
        arrays[row_id] = short
        rows.append(dict(row_id=row_id, corpus_index=index+1, system="tasep", control=case["alpha"], seed=case["seed"],
            M=32, N_state=32, N_sites=32, T=1000, view="full-state", role="evaluation",
            Q_reference=values["Q_reference"], Q_exact=values["Q_exact"], Q_window=float(short.mean()),
            master=str(path), master_sha256=identities[path.name], seal_identity=plan["seal_identity"]))
    gate = evaluate_physics(records, plan["physical_gates"])
    gate.update(seal_identity=plan["seal_identity"], seal_file_sha256=seal_file_hash,
        frozen_identity=plan["frozen_identity"], master_identities=identities,
        source_sha256=plan["source_sha256"], gate_source_sha256=digest(__file__),
        pilot_physics_evidence=plan["pilot_physics_gate"],
        interpretation="same finite-N open TASEP with fresh independent seeds; event-weighted future density, exact finite-N check, no raw-row exclusions")
    gate_text = json.dumps(gate, indent=2, allow_nan=False) + "\n"
    if gate["passes"]:
        export_arrays(output, arrays, rows, dict(system="tasep", control_label="entry rate alpha", quantity_label="particle density",
            fixed_beta=.2, N=32, thermodynamic_boundary=.2, source=SOURCE, confirmation=True,
            all_rows_evaluation=True, frozen_path=str(frozen), frozen_identity=plan["frozen_identity"],
            seal_identity=plan["seal_identity"], physics_gate_sha256=hashlib.sha256(gate_text.encode()).hexdigest(),
            system_exporter_sha256=digest(__file__)))
    else:
        output.mkdir(parents=True)
    (output / "physics-gate.json").write_text(gate_text)
    (output / "confirmation-seal.json").write_bytes(seal.read_bytes())
    pd.DataFrame(records).to_csv(output / "physics.csv", index=False)
    print(json.dumps({key: value for key, value in gate.items() if key not in ("master_identities", "frozen_identity")}, indent=2))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--frozen", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.add_argument("--pilot-physics-gate", type=Path, required=True)
    physics_parser = sub.add_parser("physics")
    physics_parser.add_argument("--index", type=int, required=True)
    physics_parser.add_argument("--root", type=Path, required=True)
    physics_parser.add_argument("--seal", type=Path, required=True)
    export_parser = sub.add_parser("gate-export")
    for name in ("physics", "frozen", "output", "seal"):
        export_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        create_seal(args.frozen, args.output, args.pilot_physics_gate)
    elif args.command == "physics":
        run_physics(args.index, args.root, args.seal)
    elif not gate_export(args.physics, args.frozen, args.output, args.seal)["passes"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
