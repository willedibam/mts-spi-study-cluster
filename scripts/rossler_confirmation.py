"""Sealed fresh-seed Rössler confirmation; no fitting or model selection.

One self-contained NPZ per physical case. The generator and control grid are
unchanged from the validated pilot; only independent seeds are added.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import rossler_phase_sync as generator

PROTOCOL = dict(dt=.01, sample_dt=.2, observation_samples=2000, burn=2000., reference=100000.)
MODEL_FILES = ("model.npz", "geometry.json", "summary.json")
MODEL_ARRAYS = ("keep", "impute", "center", "component", "score_scale", "spi_order")
PYPSI_CONFIG = Path(__file__).resolve().parents[1] / "configs/pyspi/benchmarked_p90.yaml"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cases():
    controls = list(dict.fromkeys(row["coupling"] for row in generator.pilot_cases() if row["arm"] == "primary"))
    return [dict(coupling=c, seed=s, role="evaluation") for c in controls for s in range(2609151001, 2609151033)]


def model_identity(frozen):
    identity = {name: digest(frozen / name) for name in MODEL_FILES}
    arrays = {}
    with np.load(frozen / "model.npz", allow_pickle=False) as model:
        require(set(model.files) == set(MODEL_ARRAYS), "unexpected frozen model array schema")
        for name in MODEL_ARRAYS:
            value = model[name]
            arrays[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                               sha256=hashlib.sha256(value.tobytes()).hexdigest())
    summary = json.loads((frozen / "summary.json").read_text())
    geometry = json.loads((frozen / "geometry.json").read_text())
    require(summary["passes"] and geometry["passes"], "pilot model did not pass existing gates")
    require(summary["display_sign"] in (-1, 1), "invalid frozen display orientation")
    return dict(files=identity, arrays=arrays, display_sign=summary["display_sign"])


def create_seal(frozen, output, pilot_physics_gate=None):
    if output.exists():
        raise FileExistsError(output)
    if pilot_physics_gate is None:
        pilot_physics_gate = frozen.parents[1] / "physics-analysis-verified/physics-gate.json"
    pilot_gate = json.loads(pilot_physics_gate.read_text())
    require(pilot_gate["passes"] and pilot_gate["checks"]["timestep_check"], "pilot physics/timestep validation missing")
    require(pilot_gate["source_sha256"] == digest(generator.__file__), "generator differs from validated pilot")
    plan = dict(schema="rossler-fresh-seed-confirmation-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases(), protocol=PROTOCOL, primary_M=6, primary_T=1000,
        roles="all evaluation; no development rows or refitting",
        frozen_path=str(frozen.resolve()), frozen_identity=model_identity(frozen),
        source_sha256=digest(generator.__file__), confirmation_source_sha256=digest(__file__),
        pyspi_config_sha256=digest(PYPSI_CONFIG),
        pilot_physics_gate=dict(path=str(pilot_physics_gate.resolve()), sha256=digest(pilot_physics_gate),
                               maximum_anchor_dt_difference=pilot_gate["maximum_anchor_dt_difference"]),
        physical_gates=dict(raw_failure_policy="stop without exclusions", minimum_raw_sd=1e-8,
            minimum_radius=1e-3, maximum_phase_increment=.1,
            maximum_poincare_frequency_discrepancy=2*np.pi/PROTOCOL["reference"] + 1e-9,
            maximum_half_difference_p95=.004, minimum_endpoint_contrast=.01,
            maximum_entrained_endpoint_mean=.0002),
        spi_gates="unchanged shared analysis: frozen selected-feature mask and model; <=5% selected missingness per row, <=10% excluded overall and >=75% eligible per control cell",
        inference="fresh-seed confirmation of across-control physical frequency-mismatch tracking; no new control interpolation or exact boundary claim")
    plan["seal_identity"] = canonical_hash(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(seal=str(output), seal_identity=plan["seal_identity"], cases=len(plan["cases"]))))
    return plan


def read_seal(path, frozen=None):
    plan = json.loads(path.read_text())
    unsealed = {key: value for key, value in plan.items() if key != "seal_identity"}
    require(plan["seal_identity"] == canonical_hash(unsealed), "seal contents changed")
    require(plan["cases"] == cases() and plan["protocol"] == PROTOCOL, "confirmation design changed")
    require(plan["source_sha256"] == digest(generator.__file__), "generator changed after sealing")
    require(plan["confirmation_source_sha256"] == digest(__file__), "confirmation source changed after sealing")
    require(plan["pyspi_config_sha256"] == digest(PYPSI_CONFIG), "p90 catalogue changed after sealing")
    if frozen is not None:
        require(model_identity(frozen) == plan["frozen_identity"], "frozen model, geometry, summary or sign changed")
    return plan


def run_physics(index, root, seal):
    plan = read_seal(seal)
    require(0 <= index < len(plan["cases"]), "case index outside sealed design")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case-{index:04d}.npz"
    if path.exists():
        raise FileExistsError(path)
    case = plan["cases"][index]
    arrays, metadata = generator.simulate(case["coupling"], case["seed"], **plan["protocol"])
    metadata.update(case_index=index, role="evaluation", seal_identity=plan["seal_identity"],
                    seal_file_sha256=digest(seal), frozen_model_sha256=plan["frozen_identity"]["files"]["model.npz"])
    # Exclusive-create is intentional: never overwrite a completed physical case.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays,
                            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    print(json.dumps(dict(case_index=index, path=str(path), Q=metadata["reference_summary"]["Q"],
                          elapsed_seconds=metadata["elapsed_seconds"])), flush=True)
    return path


def row_diagnostics(metadata, x):
    reference = metadata["reference_summary"]
    values = [reference["Q"], reference["PLV"], reference["Q_half_difference"],
              *reference["minimum_radius"], *reference["maximum_phase_increment"],
              *reference["poincare_frequency_discrepancy"]]
    return dict(case_index=metadata["case_index"], control=metadata["coupling"], seed=metadata["seed"],
        Q=reference["Q"], PLV=reference["PLV"], half_difference=reference["Q_half_difference"],
        minimum_radius=min(reference["minimum_radius"]), maximum_phase_increment=max(reference["maximum_phase_increment"]),
        poincare_discrepancy=max(reference["poincare_frequency_discrepancy"]),
        finite_summary=bool(np.isfinite(values).all()),
        raw_ok=bool(x.shape == (6, 2000) and np.isfinite(x).all() and np.all(x[:, :1000].std(axis=1) > 1e-8)))


def evaluate_physics(records, limits):
    frame = pd.DataFrame(records)
    means = frame.groupby("control").Q.mean()
    require(len(means) >= 2, "need both control endpoints")
    contrast = float(means.iloc[0] - means.iloc[-1])
    checks = dict(all_raw_channels_vary=bool(frame.raw_ok.all()),
        finite_physical_summaries=bool(frame.finite_summary.all()),
        phase_geometry=bool((frame.minimum_radius > limits["minimum_radius"]).all()
                            and (frame.maximum_phase_increment < limits["maximum_phase_increment"]).all()),
        poincare_agreement=bool((frame.poincare_discrepancy < limits["maximum_poincare_frequency_discrepancy"]).all()),
        reference_stability=bool(frame.half_difference.quantile(.95) < limits["maximum_half_difference_p95"]),
        sizeable_contrast=bool(contrast > limits["minimum_endpoint_contrast"]),
        entrained_endpoint=bool(means.iloc[-1] < limits["maximum_entrained_endpoint_mean"]))
    return dict(passes=all(checks.values()), checks=checks, rows=len(frame), excluded_rows=0,
        Q_contrast=contrast, half_difference_p95=float(frame.half_difference.quantile(.95)),
        control_mean_Q=means.to_dict())


def gate_export(physics, frozen, output, seal):
    from scripts.finite_regime_pipeline import export_arrays
    if output.exists():
        raise FileExistsError(output)
    plan = read_seal(seal, frozen)
    expected = [physics / f"case-{i:04d}.npz" for i in range(len(plan["cases"]))]
    require(set(physics.glob("case-*.npz")) == set(expected), "incomplete or extra confirmation physical cases")
    records, rows, arrays, identities = [], [], {}, {}
    for index, (path, case) in enumerate(zip(expected, plan["cases"])):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"]))
            x = archive["X"]
            require(meta["case_index"] == index, "case index mismatch")
            for key, value in case.items():
                require(meta[key] == value, f"case {index} identity mismatch: {key}")
            for key, value in PROTOCOL.items():
                require(meta[key] == value, f"case {index} protocol mismatch: {key}")
            require(meta["M"] == meta["N_state"] == 6 and meta["N_oscillators"] == 2, "full-state dimensions changed")
            require(meta["source_sha256"] == plan["source_sha256"], "case generator source differs")
            require(meta["seal_identity"] == plan["seal_identity"] and meta["seal_file_sha256"] == digest(seal), "case seal differs")
            require(meta["frozen_model_sha256"] == plan["frozen_identity"]["files"]["model.npz"], "case model seal differs")
            require(hashlib.sha256(x.tobytes()).hexdigest() == meta["input_sha256"], "input digest mismatch")
            # Recompute physical reference summaries from retained sufficient stats.
            stats = archive["reference_stats"]
            require(stats.shape == (10, 19), "unexpected reference-statistics shape")
            steps = int(round(PROTOCOL["reference"] / PROTOCOL["dt"]))
            recomputed = generator.summarize(generator.combine_stats(stats), steps, PROTOCOL["dt"])
            for key in ("Q", "PLV", "minimum_radius", "maximum_phase_increment", "poincare_frequency_discrepancy"):
                require(np.allclose(recomputed[key], meta["reference_summary"][key], atol=1e-12, rtol=0),
                        f"physical {key} inconsistent with sufficient statistics")
            half_Q = [generator.summarize(generator.combine_stats(part), steps // 2, PROTOCOL["dt"])["Q"]
                      for part in np.array_split(stats, 2)]
            require(np.isclose(abs(half_Q[0] - half_Q[1]), meta["reference_summary"]["Q_half_difference"],
                               atol=1e-12, rtol=0), "physical half precision inconsistent with sufficient statistics")
            records.append(row_diagnostics(meta, x))
            short = np.ascontiguousarray(x[:, :1000])
        identities[path.name] = digest(path)
        phase = np.unwrap(np.arctan2(short[[1, 4]], short[[0, 3]]), axis=1)
        phase_gain = phase[:, -1] - phase[:, 0]
        window_Q = float(abs(phase_gain[0] - phase_gain[1]) / (999 * PROTOCOL["sample_dt"]))
        name = f"rossler-c{case['coupling']:.6f}-s{case['seed']}-m6-t1000"
        arrays[name] = short
        rows.append(dict(row_id=name, corpus_index=index+1, system="rossler-phase", control=case["coupling"],
            seed=case["seed"], M=6, N_state=6, N_oscillators=2, T=1000, view="full-state", role="evaluation",
            Q_reference=meta["reference_summary"]["Q"], Q_window=window_Q,
            PLV_reference=meta["reference_summary"]["PLV"],
            master=str(path), master_sha256=identities[path.name], seal_identity=plan["seal_identity"]))
    gate = evaluate_physics(records, plan["physical_gates"])
    gate.update(system="rossler-phase", seal_identity=plan["seal_identity"], seal_file_sha256=digest(seal),
        frozen_identity=plan["frozen_identity"], master_identities=identities,
        source_sha256=plan["source_sha256"], gate_source_sha256=digest(__file__),
        timestep_evidence=plan["pilot_physics_gate"],
        interpretation="fresh independent seeds; same control grid, frozen coordinate and validated solver; no physical row exclusions")
    gate_text = json.dumps(gate, indent=2, allow_nan=False) + "\n"
    if gate["passes"]:
        export_arrays(output, arrays, rows, dict(system="rossler-phase", control_label="coupling C",
            quantity_label="mean angular-frequency mismatch", source=generator.SOURCE,
            confirmation=True, all_rows_evaluation=True, frozen_path=str(frozen),
            frozen_identity=plan["frozen_identity"], seal_identity=plan["seal_identity"],
            physics_gate_sha256=hashlib.sha256(gate_text.encode()).hexdigest(),
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
    seal_parser.add_argument("--pilot-physics-gate", type=Path)
    physics_parser = sub.add_parser("physics")
    physics_parser.add_argument("--index", type=int, required=True)
    physics_parser.add_argument("--root", type=Path, required=True)
    physics_parser.add_argument("--seal", type=Path, required=True)
    export_parser = sub.add_parser("gate-export")
    export_parser.add_argument("--physics", type=Path, required=True)
    export_parser.add_argument("--frozen", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        create_seal(args.frozen, args.output, args.pilot_physics_gate)
    elif args.command == "physics":
        run_physics(args.index, args.root, args.seal)
    elif not gate_export(args.physics, args.frozen, args.output, args.seal)["passes"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
