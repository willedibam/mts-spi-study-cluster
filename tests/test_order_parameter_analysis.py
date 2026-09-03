import json

import numpy as np
import pandas as pd

from scripts.analyze_kuramoto_order_benchmark import (
    _conditional_path_gap,
    _ci_excludes_zero,
    _curve_rmse,
    _joint_path_noise_bootstrap,
    _paired_cell_bootstrap,
    _truth_result_payload,
)
from scripts.analyze_desai_zwanzig_fine_boundary import (
    _normalized_maximum_adjacent_change,
    _paired_sharpness_bootstrap,
)
from scripts.analyze_stuart_landau_confirmation import (
    _metadata_frame,
    _steepest_interval,
)

from src.order_parameter_analysis import (
    clustered_bootstrap_difference,
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    fit_frozen_pc1,
    input_only_features,
    residualize_by_group,
)


def test_frozen_pc1_is_deterministic_and_uses_development_imputation() -> None:
    development = np.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 0.0, 1.1],
            [1.0, 0.0, 0.9],
            [2.0, 0.0, 1.2],
        ]
    )
    model = fit_frozen_pc1(development, variance_threshold=0.05)
    assert model.feature_indices.tolist() == [0, 2]
    assert model.component[np.argmax(np.abs(model.component))] > 0.0
    scores, missing = model.transform(np.array([[np.nan, 7.0, 1.05]]))
    assert scores.shape == (1,)
    assert missing.tolist() == [0.5]
    assert np.isfinite(scores[0])
    repeated_scores, _ = model.transform(np.array([[np.nan, 7.0, 1.05]]))
    np.testing.assert_array_equal(scores, repeated_scores)


def test_input_only_features_detect_coherent_channels() -> None:
    time = np.linspace(0.0, 20.0, 1000)
    coherent = np.column_stack([np.cos(time + shift) for shift in (0.0, 0.05, -0.05)])
    rng = np.random.default_rng(4)
    incoherent = np.column_stack(
        [np.cos((1.0 + 0.3 * index) * time + rng.uniform(0, 2 * np.pi)) for index in range(3)]
    )
    high = input_only_features(coherent)
    low = input_only_features(incoherent)
    assert high["mean_abs_correlation"] > low["mean_abs_correlation"]
    assert high["covariance_leading_fraction"] > low["covariance_leading_fraction"]
    assert high["analytic_phase_coherence"] > low["analytic_phase_coherence"]


def test_group_residuals_and_cluster_bootstrap() -> None:
    residuals = residualize_by_group([1.0, 3.0, 5.0, 9.0], ["a", "a", "b", "b"])
    np.testing.assert_allclose(residuals, [-1.0, 1.0, -2.0, 2.0])
    draws = clustered_bootstrap_difference(
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.1, 0.9],
        [0.5, 0.5, 0.5, 0.5],
        [0, 0, 1, 1],
        n_resamples=100,
        seed=9,
    )
    assert np.all(draws < 0.0)
    mae = clustered_bootstrap_mae(
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.1, 0.9],
        [0, 0, 1, 1],
        n_resamples=50,
        seed=3,
    )
    assert np.all((mae >= 0.0) & (mae <= 0.1))
    overall, within = clustered_bootstrap_spearman(
        [0.01, 0.69, 0.11, 0.82, 0.19, 0.91, 0.31, 0.98],
        [0.0, 0.7, 0.1, 0.8, 0.2, 0.9, 0.3, 1.0],
        ["low", "high"] * 4,
        np.repeat(np.arange(4), 2),
        n_resamples=50,
        seed=2,
    )
    assert np.nanmedian(overall) > 0.8
    assert np.nanmedian(within) > 0.8


def test_steepest_interval_localizes_absolute_change() -> None:
    curve = pd.Series([0.0, 0.1, 1.1, 1.2], index=[0.0, 0.2, 0.4, 0.6])
    result = _steepest_interval(curve)
    assert result["interval"] == [0.2, 0.4]
    assert np.isclose(result["midpoint"], 0.3)
    assert np.isclose(result["maximum_absolute_slope"], 5.0)


def test_normalized_maximum_adjacent_change_is_scale_free() -> None:
    curve = pd.Series([1.0, 0.8, 0.2, 0.0], index=[1.0, 1.1, 1.2, 1.3])
    result = _normalized_maximum_adjacent_change(curve)
    scaled = _normalized_maximum_adjacent_change(7.0 * curve - 4.0)
    assert result["interval"] == [1.1, 1.2]
    assert np.isclose(result["fraction_of_total_range"], 0.6)
    assert result["interval"] == scaled["interval"]
    assert np.isclose(
        result["fraction_of_total_range"], scaled["fraction_of_total_range"]
    )


def test_paired_sharpness_bootstrap_preserves_identical_curves() -> None:
    frame = pd.DataFrame(
        {
            "instance": np.repeat(np.arange(4), 4),
            "sigma": np.tile([1.0, 1.1, 1.2, 1.3], 4),
            "q_display": np.tile([1.0, 0.9, 0.2, 0.1], 4),
            "Q_mean_abs": np.tile([1.0, 0.9, 0.2, 0.1], 4),
        }
    )
    draws = _paired_sharpness_bootstrap(
        frame,
        np.ones(len(frame), dtype=bool),
        bootstraps=50,
        seed=71,
    )
    np.testing.assert_allclose(draws, 0.0, atol=1e-15)


def test_stuart_landau_metadata_arm_uses_labels(tmp_path) -> None:
    dataset = tmp_path / "fine-boundary-row"
    dataset.mkdir()
    metadata = {
        "mts_class": "stuart-landau-locking-boundary-confirmation",
        "labels": ["stuart-landau", "full-observation"],
        "M": 32,
        "T": 1000,
        "instance_index": 0,
        "generator": {
            "resolved_params": {
                "frequency_half_width": 0.74,
                "coupling": 0.8,
                "N_full": None,
            }
        },
        "experiment": {
            "git_commit": "abc123",
            "config_sha256": "config-hash",
            "git_dirty": False,
        },
        "pyspi": {"config_sha256": "pyspi-hash", "version": "1.0"},
    }
    (dataset / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    frame = _metadata_frame(
        {"dataset_paths": np.asarray([str(dataset)], dtype=object)}, tmp_path
    )
    assert frame.loc[0, "arm"] == "full"


def test_conditional_path_gap_recovers_zero_and_known_shift() -> None:
    grid = np.linspace(0.1, 0.9, 12)
    target = np.concatenate([grid, grid])
    first = np.arange(grid.size)
    second = np.arange(grid.size, 2 * grid.size)
    assert np.isclose(_conditional_path_gap(target, target, first, second), 0.0, atol=1e-12)
    shifted = target.copy()
    shifted[second] += 0.35
    assert np.isclose(
        _conditional_path_gap(target, shifted, first, second), 0.35, atol=1e-12
    )


def test_joint_path_noise_bootstrap_pairs_shared_gaussian_resamples() -> None:
    grid = np.linspace(0.1, 0.9, 8)
    target = np.tile(grid, 12)
    clusters = np.repeat(np.arange(12), grid.size)
    gaussian_first = np.arange(0, 4 * grid.size)
    gaussian_second = np.arange(4 * grid.size, 8 * grid.size)
    logistic = np.arange(8 * grid.size, 12 * grid.size)
    prediction = target.copy()
    prediction[gaussian_second] += 0.1
    prediction[logistic] += 0.4
    cross, noise, difference = _joint_path_noise_bootstrap(
        target,
        prediction,
        clusters,
        gaussian_first,
        gaussian_second,
        logistic,
        n_resamples=40,
        seed=17,
    )
    np.testing.assert_allclose(difference, cross - noise)
    repeated = _joint_path_noise_bootstrap(
        target,
        prediction,
        clusters,
        gaussian_first,
        gaussian_second,
        logistic,
        n_resamples=40,
        seed=17,
    )
    for observed, expected in zip((cross, noise, difference), repeated):
        np.testing.assert_array_equal(observed, expected)


def test_identical_paired_and_cell_curves_have_zero_rmse() -> None:
    groups = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    values = np.array([0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8])
    paired = np.arange(4)
    cell = np.arange(4, 8)
    clusters = np.arange(8)
    assert _curve_rmse(values, groups, paired, cell) == 0.0
    draws = _paired_cell_bootstrap(
        values,
        groups,
        clusters,
        paired,
        cell,
        n_resamples=30,
        seed=3,
    )
    np.testing.assert_allclose(draws, np.zeros_like(draws), atol=1e-15)


def test_truth_result_payload_preserves_full_and_complement_targets() -> None:
    frame = pd.DataFrame(
        {
            "r_full_future": [0.11, 0.12],
            "r_unobserved_future": [0.21, 0.22],
            "r_full": [0.31, 0.32],
            "r_unobserved": [0.41, 0.42],
            "r_observed": [0.51, 0.52],
        }
    )
    payload = _truth_result_payload(frame)
    np.testing.assert_array_equal(payload["r_full_future"], [0.11, 0.12])
    np.testing.assert_array_equal(payload["r_unobserved_future"], [0.21, 0.22])
    assert not np.array_equal(
        payload["r_full_future"], payload["r_unobserved_future"]
    )


def test_claim_interval_rule_requires_a_strict_one_sided_interval() -> None:
    assert _ci_excludes_zero([0.01, 0.20])
    assert _ci_excludes_zero([-0.20, -0.01])
    assert not _ci_excludes_zero([-0.01, 0.20])
    assert not _ci_excludes_zero([0.0, 0.20])
