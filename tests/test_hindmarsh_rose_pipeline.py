from copy import deepcopy

import numpy as np
import pytest

from scripts import hindmarsh_rose_pipeline as pipeline
from scripts.hindmarsh_rose_pipeline import branch_label, evaluate_gates, validate_summary
from scripts.hindmarsh_rose_spike_adding import burst_statistics


def gate_rows():
    def row(control, q, arm="primary", seed=1):
        return dict(control=control, Q=q, arm=arm, seed=seed, branch=f"{'five' if q == 5 else 'six'}-spike",
                    raw_ok=True, bursts=140, segmentation_ok=True, threshold_ok=True,
                    minimum_counted_peak=1., counts_allowed=True, stable=True, duty=.47)
    return [row(2.68, 6), row(2.6819, 6), row(2.688, 5),
            row(2.68, 6, "half-dt"), row(2.6819, 5, "half-dt"),
            row(2.6819, 5, "preparation"), row(2.6819, 6, "preparation")]


def test_branch_labels_and_real_reduction():
    assert branch_label({"5": 20}) == "five-spike"
    assert branch_label({"6": 20}) == "six-spike"
    assert branch_label({"5": 10, "6": 10}) == "mixed-5-6"
    dt = .01
    x = 2*np.sin(np.arange(0, 100, dt))
    summary = burst_statistics(x, dt)[0]
    assert validate_summary(summary)
    wrong = deepcopy(summary)
    wrong["Q"] += 1
    assert not validate_summary(wrong)


def test_demonstrated_coexisting_step_branches_are_not_integrator_failure():
    result = evaluate_gates(gate_rows())
    assert result["passes"]
    assert result["half_dt_branch_changes"][0]["independently_demonstrated_coexistence"]
    without_evidence = [r for r in gate_rows() if not (r["arm"] == "preparation" and r["Q"] == 5)]
    assert not evaluate_gates(without_evidence)["passes"]


def test_gate_does_not_demand_unique_control_truth_but_rejects_bad_physics():
    rows = gate_rows()
    extra = deepcopy(rows[1])
    extra.update(seed=2, Q=5, branch="five-spike")
    rows.append(extra)
    assert evaluate_gates(rows)["passes"]
    for key, value in [("raw_ok", False), ("threshold_ok", False), ("bursts", 2),
                       ("stable", False), ("minimum_counted_peak", .1)]:
        bad = deepcopy(rows)
        bad[0][key] = value
        assert not evaluate_gates(bad)["passes"]


def test_seal_rechecks_diagnostic_files_not_only_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.generator, "pilot_cases", lambda: [{}, {}])
    hashes = []
    for i in range(2):
        archive = tmp_path/f"case-{i:04d}.npz"
        metadata = archive.with_suffix(".json")
        archive.write_bytes(f"archive-{i}".encode())
        metadata.write_text("{}")
        hashes.append(dict(case_index=i, master_sha256=pipeline.digest(archive),
                           metadata_sha256=pipeline.digest(metadata)))
    gate = dict(passes=True, source_sha256=pipeline.digest(pipeline.generator.__file__),
                gate_source_sha256=pipeline.digest(pipeline.__file__), bank_hashes=hashes)
    pipeline.verify_seal(tmp_path, gate)
    (tmp_path/"case-0001.json").write_text('{"changed":"diagnostic"}')
    with pytest.raises(ValueError, match="evidence changed"):
        pipeline.verify_seal(tmp_path, gate)
