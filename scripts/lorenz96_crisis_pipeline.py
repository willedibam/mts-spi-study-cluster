"""Prospective physical eligibility and corpus export for the N=8 L96 crisis.

Freeze raw h=.25 alternating-mode residence before accessing the pilot bank.
Its physical interpretation is conditional on the explicit classifier gates below.
The partition is ours, not a claim to reproduce the paper's unspecified partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import lorenz96_crisis as generator

PRIMARY = "raw_h0.25"
THRESHOLD = .25


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sampled_switch_count(x, threshold=THRESHOLD):
    """Apply the SAME hysteresis to actual input samples, without interpolation."""
    x = np.asarray(x)
    require(x.ndim == 2 and x.shape[0] == 8 and x.shape[1] >= 2, "expected 8xT states")
    require(np.isfinite(x).all(), "nonfinite states")
    mode = (x[::2].sum(axis=0) - x[1::2].sum(axis=0)) / 8
    label, count = 0, 0
    for value in mode:
        following = 1 if value > threshold else -1 if value < -threshold else label
        count += int(label != 0 and following != label)
        label = following
    return count


def count_compatible(a, b, *, relative=.2):
    """Prospective tolerance: 20% of mean count or three Poisson-scale SDs.

    This is an operational stability gate, not a Poisson-process assumption or CI.
    Block variability remains reported because crises can be overdispersed.
    """
    return abs(a - b) <= max(relative * .5 * (a + b), 3 * np.sqrt(a + b))


def evaluate_gates(records):
    frame = pd.DataFrame(records)
    primary = frame[frame.arm == "primary"]
    pre = primary[np.isclose(primary.control, -6.4, rtol=0, atol=1e-12)]
    post = primary[np.isclose(primary.control, -6.6, rtol=0, atol=1e-12)]
    require(len(pre) > 0 and len(post) > 0, "missing fixed endpoint anchors")
    timestep = []
    for row in frame[frame.arm == "half-dt"].itertuples():
        peer = primary[(primary.seed == row.seed) & np.isclose(primary.control, row.control, rtol=0, atol=1e-12)]
        require(len(peer) == 1, "missing/unmatched half-dt peer")
        timestep.append(count_compatible(float(peer.switch_count.iloc[0]), row.switch_count))
    require(len(timestep) > 0, "no timestep anchors")
    between = [count_compatible(float(row.switch_count), float(post.switch_count.mean()))
               for row in post.itertuples()]
    checks = dict(all_raw_channels_vary=bool(frame.raw_ok.all()),
        positive_reference_lyapunov=bool(np.isfinite(frame.largest_lyapunov).all() and (frame.largest_lyapunov > 0).all()),
        precrisis_anchor_no_switches=bool((pre.switch_count == 0).all() and (pre.variant_max_count == 0).all()),
        precrisis_partition_separated=bool(pre.mode_away_from_deadband.all()),
        postcrisis_anchor_well_sampled=bool((post.switch_count >= 50).all()),
        postcrisis_both_regions_visited=bool((post.minimum_occupancy > .05).all()),
        classifier_threshold_smoothing_robust=bool(frame.classifier_robust.all()),
        retained_anchor_sampling_robust=bool(frame.sampled_anchor_ok.all()),
        postcrisis_reference_halves_compatible=bool(post.halves_compatible.all()),
        postcrisis_seeds_compatible=bool(all(between)),
        timestep_counts_compatible=bool(all(timestep)),
        residence_censoring_consistent=bool(frame.censoring_ok.all()))
    return dict(passes=all(checks.values()), checks=checks,
                postcrisis_switch_counts=post.switch_count.astype(int).tolist(),
                control_mean_Q=primary.groupby("control").Q.mean().to_dict())


def validate_residence(row, duration):
    count = int(row["switch_count"])
    completed = sum(row["completed_count_by_state"])
    total = sum(row["completed_duration_by_state"])
    blocks = np.asarray(row["switch_rate_blocks"]) * duration / len(row["switch_rate_blocks"])
    correct = (count >= 0 and completed == max(0, count - 1)
        and np.isclose(row["switch_rate"], count / duration, rtol=0, atol=1e-14)
        and np.isclose(blocks.sum(), count, rtol=0, atol=1e-7)
        and np.all(blocks >= 0) and np.allclose(blocks, np.rint(blocks), atol=1e-7)
        and bool(row["no_switch_window_censored"]) == (count == 0)
        and bool(row["first_residence_left_censored"])
        and 0 <= row["last_residence_right_censored_duration"] <= duration
        and 0 <= row["unknown_duration"] <= duration
        and np.isclose(sum(row["occupancy_fraction_by_state"]) + row["unknown_duration"] / duration, 1., atol=1e-8))
    if completed:
        correct = correct and total > 0 and np.isclose(row["mean_completed_residence"], total / completed)
        correct = correct and np.isclose(row["reciprocal_completed_residence"], completed / total)
    else:
        correct = correct and row["mean_completed_residence"] is None and row["reciprocal_completed_residence"] is None
    return bool(correct)


def inspect_bank(physics):
    expected = generator.pilot_cases()
    paths = [physics / f"case-{i:04d}.json" for i in range(len(expected))]
    require(set(physics.glob("case-*.json")) == set(paths), "bank must contain exactly 171 expected JSON cases")
    require(set(physics.glob("case-*.npz")) == {p.with_suffix(".npz") for p in paths}, "bank archive identities incomplete/extra")
    source_hash = digest(generator.__file__)
    records, hashes = [], []
    for index, (path, case) in enumerate(zip(paths, expected)):
        meta = json.loads(path.read_text())
        require(meta["case_index"] == index, "case index mismatch")
        for key, value in case.items():
            correct = np.isclose(meta[key], value, rtol=0, atol=1e-12) if isinstance(value, float) else meta[key] == value
            require(correct, f"case {index} parameter mismatch: {key}")
        require(meta["source_sha256"] == source_hash, f"case {index} generator source differs from inspected source")
        require(meta["burn"] == 10000 and meta["reference"] == 100000 and meta["blocks"] == 10,
                "unplanned burn/reference/block protocol")
        require(meta["sample_dt"] == .1 and meta["observation_samples"] == 2000 and meta["M"] == meta["N"] == 8,
                "unplanned observation protocol")
        require(meta["mode_definition"] == "A=(sum_even_zero_based(x)-sum_odd_zero_based(x))/8", "mode definition differs")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            x = archive["X"]
            require(x.shape == (8, 2000), "wrong full-state input shape")
            require(hashlib.sha256(x.tobytes()).hexdigest() == meta["input_sha256"], "input hash mismatch")
            raw_ok = bool(np.isfinite(x).all() and np.all(x[:, :1000].std(axis=1) > 1e-8))
            sampled_ok = True
            if case["anchor_duration"]:
                anchor = archive["anchor_X"]
                require(anchor.shape == (8, 100000), "wrong anchor shape")
                sampled = sampled_switch_count(anchor)
                # Stored anchor spans first reference block, except its t=0 point.
                dense = meta["residence_diagnostics"][PRIMARY]["switch_rate_blocks"][0] * 10000
                sampled_ok = abs(sampled - dense) <= 1
        rows = meta["residence_diagnostics"]
        require(set(rows) == set(generator.VARIANTS), "classifier variants missing/extra")
        row = rows[PRIMARY]
        count = row["switch_count"]
        counts = [r["switch_count"] for r in rows.values()]
        block_counts = np.asarray(row["switch_rate_blocks"]) * 10000
        low, high = min(meta["mode_block_min"]), max(meta["mode_block_max"])
        records.append(dict(case_index=index, arm=case["arm"], seed=case["seed"], control=case["forcing"],
            dt=case["dt"], Q=count / meta["reference"], switch_count=count,
            raw_ok=raw_ok, largest_lyapunov=meta["largest_lyapunov"],
            variant_min_count=min(counts), variant_max_count=max(counts),
            classifier_robust=max(counts) - min(counts) <= max(2, .05 * max(counts)),
            sampled_anchor_ok=sampled_ok, mode_min=low, mode_max=high,
            mode_away_from_deadband=low > .5 or high < -.5,
            minimum_occupancy=min(row["occupancy_fraction_by_state"]),
            first_half_count=float(block_counts[:5].sum()), second_half_count=float(block_counts[5:].sum()),
            halves_compatible=count_compatible(float(block_counts[:5].sum()), float(block_counts[5:].sum())),
            censoring_ok=all(validate_residence(r, meta["reference"]) for r in rows.values())))
        hashes.append(dict(case_index=index, metadata_sha256=digest(path), master_sha256=digest(path.with_suffix(".npz"))))
    return records, hashes, source_hash


def physics_gate(physics, output):
    records, hashes, source_hash = inspect_bank(physics)
    result = evaluate_gates(records)
    result.update(system="lorenz96-crisis", records=len(records), primary_records=168,
        primary_classifier=PRIMARY, primary_threshold=THRESHOLD,
        Q_definition="finite-reference switch count / physical reference duration",
        partition_definition="hysteretic sign of alternating spatial mode, fixed h=.25",
        partition_attribution="our operational realization of published precrisis-region residence; validated by prespecified anchors",
        source=generator.SOURCE, published_crisis_approx=-6.4717,
        inference_limit="finite-time switching-rate inference; no exact crisis proof or discontinuous rate claimed",
        near_crisis_policy="rare or zero events near the boundary are reported as censored/uncertain, not automatically excluded",
        compatibility_rule="count difference <= max(20% of mean count, 3*sqrt(total count)); operational tolerance, not a claimed Poisson CI",
        source_sha256=source_hash, gate_source_sha256=digest(__file__), bank_hashes=hashes)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(output / "physics.csv", index=False)
    (output / "physics-gate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "bank_hashes"}, indent=2))
    return result


def export(physics, gate_dir, output):
    # Lazy import: physics gates do not depend on sklearn or the shared exporter.
    from scripts.finite_regime_pipeline import export_arrays
    gate_path = gate_dir / "physics-gate.json"
    gate = json.loads(gate_path.read_text())
    require(gate["passes"], "physical/classifier gates failed; no p90 corpus exported")
    require(gate["primary_classifier"] == PRIMARY and gate["primary_threshold"] == THRESHOLD,
            "classifier differs from frozen exporter")
    require(gate["gate_source_sha256"] == digest(__file__), "gate source changed since validation")
    require(gate["source_sha256"] == digest(generator.__file__), "generator changed since validation")
    require(len(gate["bank_hashes"]) == len(generator.pilot_cases()), "incomplete gate identities")
    for index, sealed in enumerate(gate["bank_hashes"]):
        path = physics / f"case-{index:04d}.npz"
        require(sealed["case_index"] == index and digest(path) == sealed["master_sha256"]
                and digest(path.with_suffix(".json")) == sealed["metadata_sha256"],
                "physics bank or half-dt evidence changed since gate")
    rows, arrays = [], {}
    expected = generator.pilot_cases()[:168]
    seeds = sorted({c["seed"] for c in expected})
    development = set(seeds[:4])
    for index, case in enumerate(expected):
        path = physics / f"case-{index:04d}.npz"
        metadata_path = path.with_suffix(".json")
        sealed = gate["bank_hashes"][index]
        require(sealed["case_index"] == index and digest(path) == sealed["master_sha256"]
                and digest(metadata_path) == sealed["metadata_sha256"], "bank changed since physical validation")
        meta = json.loads(metadata_path.read_text())
        with np.load(path, allow_pickle=False) as archive:
            x = np.ascontiguousarray(archive["X"][:, :1000])
        require(x.shape == (8, 1000) and np.isfinite(x).all() and np.all(x.std(axis=1) > 1e-8), "invalid raw input")
        input_count = sampled_switch_count(x)
        row_id = f"lorenz96-f{case['forcing']:.5f}-s{case['seed']}-m8-t1000"
        arrays[row_id] = x
        rows.append(dict(row_id=row_id, corpus_index=index + 1, system="lorenz96-crisis",
            control=case["forcing"], seed=case["seed"], M=8, N_state=8, T=1000, view="full-state",
            role="development" if case["seed"] in development else "evaluation",
            Q_reference=meta["residence_diagnostics"][PRIMARY]["switch_rate"],
            Q_window=input_count / (999 * meta["sample_dt"]), Q_window_switch_count=input_count,
            Q_window_definition="same h=.25 hysteresis on actual sampled input, divided by (T-1)*sample_dt; not a dense-time event count",
            largest_lyapunov_reference=meta["largest_lyapunov"],
            master=str(path), master_sha256=sealed["master_sha256"], metadata_sha256=sealed["metadata_sha256"]))
    require(len(rows) == 168 and sum(r["role"] == "development" for r in rows) == 84, "incorrect seed split")
    export_arrays(output, arrays, rows, dict(system="lorenz96-crisis", control_label="forcing F",
        quantity_label="inter-region switching rate", source=generator.SOURCE,
        published_crisis_approx=-6.4717, primary_classifier=PRIMARY,
        partition_attribution="ours; validated realization of published residence-time diagnostic",
        physics_gate_sha256=digest(gate_path), system_exporter_sha256=digest(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("physics", "export"):
        command = sub.add_parser(name)
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
