import numpy as np
import pytest

from scripts.lorenz96_crisis_pipeline import (count_compatible, evaluate_gates,
    sampled_switch_count, validate_residence)


def record(control, seed, count, arm="primary"):
    return dict(control=control, seed=seed, switch_count=count, Q=count/100000,
        arm=arm, raw_ok=True, largest_lyapunov=.2, variant_max_count=count,
        mode_away_from_deadband=count == 0, minimum_occupancy=.4 if count else 0,
        classifier_robust=True, sampled_anchor_ok=True, halves_compatible=True,
        censoring_ok=True)


def bank():
    return [record(-6.4, 1, 0), record(-6.4, 2, 0), record(-6.6, 1, 500),
            record(-6.6, 2, 520), record(-6.48, 1, 0),
            record(-6.6, 1, 510, "half-dt"), record(-6.4, 1, 0, "half-dt")]


def test_gate_accepts_robust_anchors_and_censored_near_boundary():
    assert evaluate_gates(bank())["passes"]
    rows = bank()
    rows[0]["switch_count"] = 1
    assert not evaluate_gates(rows)["checks"]["precrisis_anchor_no_switches"]


@pytest.mark.parametrize("field", ["raw_ok", "classifier_robust", "sampled_anchor_ok", "censoring_ok"])
def test_gate_does_not_relax_material_failures(field):
    rows = bank()
    rows[2][field] = False
    assert not evaluate_gates(rows)["passes"]


def test_counts_tolerance_and_timestep_failure():
    assert count_compatible(500, 530)
    assert not count_compatible(0, 500)
    rows = bank()
    rows[5]["switch_count"] = 1000
    assert not evaluate_gates(rows)["checks"]["timestep_counts_compatible"]


def test_actual_input_classifier_hysteresis_and_permutation_sign():
    mode = np.array([1., .1, -.1, -1., -.1, .1, 1.])
    x = np.outer(np.array([1., -1.] * 4), mode)
    assert sampled_switch_count(x) == 2
    assert sampled_switch_count(np.roll(x, 1, axis=0)) == 2
    assert sampled_switch_count(np.ones((8, 20))) == 0


def test_zero_event_residence_must_remain_censored():
    row = dict(switch_count=0, completed_count_by_state=[0, 0], completed_duration_by_state=[0., 0.],
        switch_rate_blocks=[0.] * 10, switch_rate=0., no_switch_window_censored=True,
        first_residence_left_censored=True, last_residence_right_censored_duration=100000.,
        unknown_duration=0., occupancy_fraction_by_state=[1., 0.],
        mean_completed_residence=None, reciprocal_completed_residence=None)
    assert validate_residence(row, 100000.)
    row["mean_completed_residence"] = 100000.
    assert not validate_residence(row, 100000.)
