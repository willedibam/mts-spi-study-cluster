"""Sealed physical gates and full-state corpus export for HR spike deletion.

Independent starts are primary. Up/down preparation audits remain separate;
coexisting stable five/six-spike solutions are not averaged into control-level
truth. Q_reference is each run's own future mean number of large spikes/burst.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import hindmarsh_rose_spike_adding as generator


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def branch_label(histogram):
    support = sorted(int(k) for k, v in histogram.items() if int(v) > 0)
    if support == [5]:
        return "five-spike"
    if support == [6]:
        return "six-spike"
    return "mixed-" + "-".join(map(str, support)) if support else "unclassified"


def validate_summary(summary):
    counts = np.asarray(summary["spike_counts"], dtype=float)
    histogram = {str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    return bool(len(counts) == summary["complete_bursts"] and len(counts) > 0
                and np.isfinite(counts).all() and np.all(counts == np.rint(counts))
                and histogram == summary["spike_count_histogram"]
                and summary["Q"] is not None and np.isclose(counts.mean(), summary["Q"], atol=1e-12, rtol=0)
                and len(summary["duty_cycles"]) == len(counts)
                and len(summary["all_local_maxima_counts"]) == len(counts)
                and np.all(np.asarray(summary["all_local_maxima_counts"]) >= counts))


def evaluate_gates(records):
    frame = pd.DataFrame(records)
    primary = frame[frame.arm == "primary"]
    low = primary[np.isclose(primary.control, generator.CONTROLS[0], rtol=0, atol=1e-12)]
    high = primary[np.isclose(primary.control, generator.CONTROLS[-1], rtol=0, atol=1e-12)]
    require(len(low) > 0 and len(high) > 0, "missing source endpoint anchors")
    dt_checks = []
    different_branches = []
    for row in frame[frame.arm == "half-dt"].itertuples():
        peer = primary[(primary.seed == row.seed) & np.isclose(primary.control, row.control, rtol=0, atol=1e-12)]
        require(len(peer) == 1, "missing/unmatched half-dt peer")
        peer = peer.iloc[0]
        if peer.branch == row.branch:
            dt_checks.append(abs(peer.Q-row.Q) <= .05 and abs(peer.duty-row.duty) <= .02)
        else:
            preparation = frame[(frame.arm == "preparation") & np.isclose(frame.control, row.control, rtol=0, atol=1e-12)]
            demonstrated = set(preparation.loc[preparation.stable, "branch"])
            valid = {peer.branch, row.branch} <= demonstrated and {peer.branch, row.branch} <= {"five-spike", "six-spike"}
            dt_checks.append(valid)
            different_branches.append(dict(control=float(row.control), seed=int(row.seed),
                primary_branch=peer.branch, half_dt_branch=row.branch,
                independently_demonstrated_coexistence=bool(valid)))
    require(len(dt_checks) > 0, "no timestep anchors")
    checks = dict(all_raw_channels_vary=bool(frame.raw_ok.all()),
        complete_bursts_well_sampled=bool((frame.bursts >= 50).all()),
        valid_segmentation=bool(frame.segmentation_ok.all()),
        threshold_robust=bool(frame.threshold_ok.all()),
        counted_peaks_above_threshold=bool((frame.minimum_counted_peak > .5).all()),
        all_counts_within_source_five_six_regimes=bool(frame.counts_allowed.all()),
        future_reference_stability=bool(frame.stable.all()),
        source_six_spike_low_endpoint=bool((low.branch == "six-spike").mean() >= .875),
        source_five_spike_high_endpoint=bool((high.branch == "five-spike").mean() >= .875),
        timestep_same_branch_or_demonstrated_coexistence=bool(all(dt_checks)))
    return dict(passes=all(checks.values()), checks=checks,
                half_dt_branch_changes=different_branches,
                primary_branch_counts=primary.branch.value_counts().to_dict(),
                policy="no unique Q(control) required; never pool preparation branches into physical truth")


def inspect_bank(physics):
    expected = generator.pilot_cases()
    paths = [physics/f"case-{i:04d}.json" for i in range(len(expected))]
    require(set(physics.glob("case-*.json")) == set(paths), "exactly 255 expected metadata files required")
    require(set(physics.glob("case-*.npz")) == {p.with_suffix(".npz") for p in paths}, "missing/extra physics archives")
    source_hash = digest(generator.__file__)
    records, hashes = [], []
    for index, (path, case) in enumerate(zip(paths, expected)):
        meta = json.loads(path.read_text())
        require(meta["case_index"] == index, "case index mismatch")
        for key, value in case.items():
            correct = np.isclose(meta[key], value, atol=1e-12, rtol=0) if isinstance(value, float) else meta[key] == value
            require(correct, f"case {index}: parameter {key} differs")
        require(meta["source_sha256"] == source_hash, "generator source mismatch")
        require(meta["parameters"] == generator.PARAMETERS and meta["source"] == generator.SOURCE, "equation/source mismatch")
        require(meta["burn"] == 5000 and meta["reference"] == 20000 and meta["continuation_dwell"] == 1000,
                "unplanned physical durations")
        require(meta["M"] == meta["N_state"] == 3 and meta["N_neurons"] == 1
                and meta["observation_samples"] == 4000 and meta["sample_dt"] == .25
                and meta["diagnostic_dt"] == .05, "unplanned observation protocol")
        archive_path = path.with_suffix(".npz")
        require(digest(archive_path) == meta["archive_sha256"], "archive checksum mismatch")
        with np.load(archive_path, allow_pickle=False) as archive:
            x = archive["X"]
            require(x.shape == (3, 4000), "wrong full-state shape")
            require(hashlib.sha256(x.tobytes()).hexdigest() == meta["input_sha256"], "input hash mismatch")
            embedded = json.loads(str(archive["metadata_json"]))
            require(embedded == {k: v for k, v in meta.items() if k != "archive_sha256"}, "embedded metadata differs")
            raw_ok = bool(np.isfinite(x).all() and np.all(x[:, :1000].std(axis=1) > 1e-8))
        summary = meta["reference_summary"]
        require(validate_summary(summary) and all(validate_summary(h) for h in summary["halves"]), "inconsistent burst summary")
        require(summary["spike_threshold"] == 0 and summary["separator"] == -1.2, "primary classifier changed")
        counts = np.asarray(summary["spike_counts"])
        duties = summary["duty_cycles"]
        duty_ok = all(d is not None and 0 < d < 1 for d in duties)
        sens = summary["threshold_sensitivities"]
        require([(s["spike_threshold"], s["separator"]) for s in sens]
                == [(-.5, -1.2), (.5, -1.2), (0., -1.), (0., -1.5)], "classifier sensitivities changed")
        robust = bool(summary["threshold_agreement"] and all(validate_summary(s) and s["Q"] == summary["Q"]
                      and s["valid_burst_segmentation"] for s in sens))
        half_difference = abs(summary["halves"][0]["Q"]-summary["halves"][1]["Q"])
        require(np.isclose(half_difference, summary["Q_half_difference"], atol=1e-12, rtol=0), "half reduction mismatch")
        records.append(dict(case_index=index, arm=case["arm"], seed=case["seed"],
            control=case["b"], preparation=case["preparation"], dt=case["dt"], Q=summary["Q"],
            branch=branch_label(summary["spike_count_histogram"]),
            histogram_json=json.dumps(summary["spike_count_histogram"]),
            bursts=summary["complete_bursts"], duty=summary["mean_duty_cycle"],
            raw_ok=raw_ok, segmentation_ok=bool(summary["valid_burst_segmentation"] and duty_ok),
            threshold_ok=robust, minimum_counted_peak=summary["minimum_counted_peak"],
            counts_allowed=bool(np.isin(counts, [5, 6]).all()), half_difference=half_difference,
            stable=bool(half_difference <= .1 and all(h["complete_bursts"] >= 20 for h in summary["halves"]))))
        hashes.append(dict(case_index=index, master_sha256=digest(archive_path), metadata_sha256=digest(path)))
    return records, hashes, source_hash


def physics_gate(physics, output):
    records, hashes, source_hash = inspect_bank(physics)
    result = evaluate_gates(records)
    result.update(system="hindmarsh-rose-spike-adding", records=len(records), primary_records=168,
        Q_definition="per-run future mean count of voltage maxima x>0 between complete upward x=-1.2 burst crossings",
        source=generator.SOURCE, source_sha256=source_hash,
        gate_source_sha256=digest(__file__), bank_hashes=hashes,
        initial_condition_attribution="our explicit starts/preparations; Fig6 does not identify its initial-condition history",
        limitation="three physical channels, only three undirected MPI edges; no channel augmentation or guaranteed SPI geometry")
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(output/"physics.csv", index=False)
    (output/"physics-gate.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "bank_hashes"}, indent=2))
    return result


def verify_seal(physics, gate):
    require(gate["passes"], "physics failed; no corpus exported")
    require(gate["gate_source_sha256"] == digest(__file__), "gate source changed")
    require(gate["source_sha256"] == digest(generator.__file__), "generator source changed")
    expected = generator.pilot_cases()
    require(len(gate["bank_hashes"]) == len(expected), "incomplete bank seal")
    require(len(list(physics.glob("case-*.json"))) == len(expected)
            and len(list(physics.glob("case-*.npz"))) == len(expected), "bank membership changed")
    for index, sealed in enumerate(gate["bank_hashes"]):
        path = physics/f"case-{index:04d}.npz"
        require(sealed["case_index"] == index and digest(path) == sealed["master_sha256"]
                and digest(path.with_suffix(".json")) == sealed["metadata_sha256"],
                "physics or diagnostic evidence changed after seal")


def export(physics, gate_dir, output):
    from scripts.finite_regime_pipeline import export_arrays
    gate_path = gate_dir/"physics-gate.json"
    gate = json.loads(gate_path.read_text())
    verify_seal(physics, gate)
    arrays, rows = {}, []
    expected = generator.pilot_cases()[:168]
    development = set(sorted({c["seed"] for c in expected})[:4])
    for index, case in enumerate(expected):
        path = physics/f"case-{index:04d}.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as archive:
            x = np.ascontiguousarray(archive["X"][:, :1000])
        require(x.shape == (3, 1000) and np.isfinite(x).all() and np.all(x.std(axis=1) > 1e-8), "invalid raw input")
        window, _, _ = generator.burst_statistics(x[0], meta["sample_dt"])
        row_id = f"hindmarsh-rose-b{case['b']:.7f}-s{case['seed']}-m3-t1000"
        arrays[row_id] = x
        ref = meta["reference_summary"]
        rows.append(dict(row_id=row_id, corpus_index=index+1, system="hindmarsh-rose-spike-adding",
            control=case["b"], seed=case["seed"], M=3, N_state=3, N_neurons=1, T=1000,
            view="full-state", role="development" if case["seed"] in development else "evaluation",
            preparation=case["preparation"], branch_reference=branch_label(ref["spike_count_histogram"]),
            spike_count_histogram_reference=ref["spike_count_histogram"], Q_reference=ref["Q"],
            Q_window=window["Q"], Q_window_complete_bursts=window["complete_bursts"],
            Q_window_definition="same classifier on actual first1000 sampled states; None if no complete bursts",
            duty_cycle_reference=ref["mean_duty_cycle"], master=str(path),
            master_sha256=gate["bank_hashes"][index]["master_sha256"],
            metadata_sha256=gate["bank_hashes"][index]["metadata_sha256"]))
    require(len(rows) == 168 and sum(r["role"] == "development" for r in rows) == 84, "incorrect seed split")
    export_arrays(output, arrays, rows, dict(system="hindmarsh-rose-spike-adding", control_label="intrinsic parameter b",
        quantity_label="large spikes per burst", source=generator.SOURCE,
        physical_truth="per-run future reference, never a control-averaged or branch-pooled target",
        physics_gate_sha256=digest(gate_path), system_exporter_sha256=digest(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("physics", "export"):
        command = commands.add_parser(name)
        command.add_argument("--physics", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "export":
            command.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "physics":
        physics_gate(args.physics, args.output)
    else:
        export(args.physics, args.gate_dir, args.output)


if __name__ == "__main__":
    main()
