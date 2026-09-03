from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.generators.order_parameter import (
    generate_desai_zwanzig,
    generate_kuramoto_order_parameter,
    generate_miller_huse,
    generate_stuart_landau,
    kuramoto_critical_coupling,
    miller_huse_map,
)
from src.mapping import DatasetMapping, ExperimentConfig, _derive_dataset_seed
from src.run_experiments import (
    _build_metadata,
    _desai_zwanzig_semantics,
    _ground_truth_descriptor,
    _kuramoto_semantics,
    _miller_huse_semantics,
    _stuart_landau_semantics,
    generate_synthetic_from_spec,
)


def test_ground_truth_descriptor_skips_empty_future_arrays(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.npz"
    np.savez_compressed(
        path,
        magnetization=np.asarray([0.25, -0.5]),
        magnetization_future=np.asarray([], dtype=np.float32),
    )
    descriptor = _ground_truth_descriptor(path)
    assert descriptor["magnetization_mean"] == -0.125
    assert "magnetization_future_mean" not in descriptor
    assert descriptor["arrays"]["magnetization_future"] == [0]


def test_supported_kuramoto_critical_couplings() -> None:
    assert np.isclose(kuramoto_critical_coupling("gaussian"), np.sqrt(8.0 / np.pi))
    assert np.isclose(
        kuramoto_critical_coupling("logistic"), 8.0 * np.sqrt(3.0) / np.pi**2
    )


def test_kuramoto_views_are_nested_and_ground_truth_is_hidden() -> None:
    common = dict(
        K=1.4,
        N_full=32,
        dt=0.02,
        sample_dt=0.1,
        burn_time=2.0,
        frequency_distribution="gaussian",
        zscore=False,
        return_internals=True,
    )
    short, short_info = generate_kuramoto_order_parameter(
        M=4, T=20, rng=np.random.default_rng(7), **common
    )
    long, long_info = generate_kuramoto_order_parameter(
        M=8, T=30, rng=np.random.default_rng(7), **common
    )
    np.testing.assert_allclose(short, long[:20, :4], rtol=0, atol=0)
    np.testing.assert_array_equal(
        short_info.observation_indices, long_info.observation_indices[:4]
    )
    np.testing.assert_allclose(short_info.frequencies, long_info.frequencies, rtol=0, atol=0)
    assert short.shape == (20, 4)
    assert short_info.full_phases.shape == (20, 32)
    assert short_info.r_full.shape == (20,)
    assert short_info.r_observed.shape == (20,)
    assert short_info.r_unobserved.shape == (20,)
    assert np.all((short_info.r_full >= 0.0) & (short_info.r_full <= 1.0))
    assert np.all((short_info.r_observed >= 0.0) & (short_info.r_observed <= 1.0))
    assert np.all((short_info.r_unobserved >= 0.0) & (short_info.r_unobserved <= 1.0))


def test_kuramoto_generator_reproduces_basic_synchronization_contrast() -> None:
    common = dict(
        M=16,
        T=300,
        N_full=128,
        burn_time=50.0,
        rng=np.random.default_rng(11),
        return_internals=True,
    )
    _, below = generate_kuramoto_order_parameter(K=0.8, **common)
    common["rng"] = np.random.default_rng(11)
    _, above = generate_kuramoto_order_parameter(K=2.4, **common)
    assert below.r_full.mean() < 0.3
    assert above.r_full.mean() > 0.7


def test_kuramoto_future_truth_is_not_exposed_in_the_mts() -> None:
    common = dict(
        M=4,
        T=20,
        N_full=24,
        burn_time=2.0,
        return_internals=True,
    )
    current_only, current_info = generate_kuramoto_order_parameter(
        future_truth_T=0, rng=np.random.default_rng(19), **common
    )
    observed, info = generate_kuramoto_order_parameter(
        future_truth_T=15, rng=np.random.default_rng(19), **common
    )
    assert observed.shape == (20, 4)
    assert info.full_phases.shape == (20, 24)
    assert info.r_full.shape == (20,)
    assert info.r_full_future.shape == (15,)
    assert info.r_unobserved_future.shape == (15,)
    np.testing.assert_array_equal(observed, current_only)
    np.testing.assert_array_equal(info.r_full, current_info.r_full)
    np.testing.assert_array_equal(info.r_unobserved, current_info.r_unobserved)


def test_experiment_harness_can_omit_full_phase_movie(tmp_path: Path) -> None:
    config_path = tmp_path / "compact.yaml"
    config_path.write_text(
        """
base_output_dir: data/test-compact
pyspi_config: configs/pyspi/test.yaml
defaults: {instances: 1}
mts_classes:
  - name: compact
    generator: kuramoto_order_parameter
    M_values: [4]
    T_values: [20]
    base_params:
      N_full: 24
      burn_time: 2.0
      future_truth_T: 5
      store_full_phases: false
""",
        encoding="utf-8",
    )
    spec = DatasetMapping(ExperimentConfig.from_file(config_path)).specs[0]
    observed, extras = generate_synthetic_from_spec(spec)
    assert observed.shape == (20, 4)
    assert "full_phases" not in extras["_ground_truth"]
    assert extras["_ground_truth"]["r_full_future"].shape == (5,)
    assert extras["resolved_params"]["store_full_phases"] is False
    semantics = _kuramoto_semantics(spec, extras)
    assert semantics["control"]["reduced_name"] == "kappa"
    assert np.isclose(
        semantics["control"]["reduced_value"],
        spec.generator_params.get("K", np.sqrt(8.0 / np.pi))
        / np.sqrt(8.0 / np.pi),
    )
    assert semantics["order_parameter"]["primary_analysis_array"] == "r_full_future"
    assert semantics["order_parameter"]["included_in_timeseries_input"] is False
    meta = _build_metadata(
        spec=spec,
        result=SimpleNamespace(metadata=[], errors={}),
        paths={},
        compute_seconds=1.0,
        gen_extras={
            "resolved_params": extras["resolved_params"],
            "ground_truth": {"critical_coupling": np.sqrt(8.0 / np.pi)},
        },
        experiment_provenance={
            "config": "compact.yaml",
            "config_sha256": "abc",
            "git_commit": "def",
            "git_dirty": False,
        },
    )
    assert meta["sampling_design"]["seed_scope"] == "dataset"
    assert meta["sampling_design"]["seed_group_id"] == spec.name
    assert meta["experiment"]["config_sha256"] == "abc"
    assert meta["generator"]["control"]["reduced_name"] == "kappa"


def test_miller_huse_map_and_patch_shapes() -> None:
    values = np.linspace(-1.0, 1.0, 101)
    mapped = miller_huse_map(values)
    np.testing.assert_allclose(miller_huse_map(-values), -mapped, atol=1e-14)
    assert np.max(np.abs(mapped)) <= 1.0 + 1e-14
    mapped_19 = miller_huse_map(values, mu=1.9)
    np.testing.assert_allclose(miller_huse_map(-values, mu=1.9), -mapped_19, atol=1e-14)

    observed, internals = generate_miller_huse(
        M=20,
        T=30,
        coupling=0.2,
        lattice_side=16,
        transients=100,
        future_truth_T=20,
        rng=np.random.default_rng(3),
        return_internals=True,
    )
    assert observed.shape == (30, 20)
    assert internals.full_field is None
    assert internals.patch_indices.shape == (20, 2)
    assert internals.spin_magnetization_future.shape == (20,)
    assert internals.spin_magnetization_unobserved_future.shape == (20,)
    assert np.all(np.abs(internals.spin_magnetization) <= 1.0)


def test_miller_huse_future_truth_is_not_exposed() -> None:
    common = dict(
        M=20,
        T=25,
        coupling=0.205,
        lattice_side=12,
        transients=50,
        return_internals=True,
    )
    current, current_info = generate_miller_huse(
        future_truth_T=0, rng=np.random.default_rng(23), **common
    )
    observed, info = generate_miller_huse(
        future_truth_T=30, rng=np.random.default_rng(23), **common
    )
    np.testing.assert_array_equal(observed, current)
    np.testing.assert_array_equal(info.spin_magnetization, current_info.spin_magnetization)
    assert info.spin_magnetization_future.shape == (30,)


def test_miller_huse_distributed_observations_are_nested() -> None:
    common = dict(
        T=20,
        coupling=0.205,
        lattice_side=12,
        transients=50,
        observation_mode="distributed",
        return_internals=True,
    )
    small, small_info = generate_miller_huse(
        M=8, rng=np.random.default_rng(29), **common
    )
    large, large_info = generate_miller_huse(
        M=16, rng=np.random.default_rng(29), **common
    )
    np.testing.assert_array_equal(small, large[:, :8])
    np.testing.assert_array_equal(
        small_info.patch_indices, large_info.patch_indices[:8]
    )
    np.testing.assert_array_equal(small_info.final_field, large_info.final_field)


def test_miller_huse_prefixes_share_fixed_future_truth() -> None:
    common = dict(
        M=8,
        coupling=0.205,
        lattice_side=12,
        transients=50,
        observation_mode="distributed",
        truth_start_T=40,
        future_truth_T=30,
        return_internals=True,
    )
    short, short_info = generate_miller_huse(
        T=20, rng=np.random.default_rng(37), **common
    )
    long, long_info = generate_miller_huse(
        T=40, rng=np.random.default_rng(37), **common
    )
    np.testing.assert_array_equal(short, long[:20])
    np.testing.assert_array_equal(
        short_info.spin_magnetization_future,
        long_info.spin_magnetization_future,
    )


def test_spin_generator_harness_records_compact_truth_and_semantics(tmp_path: Path) -> None:
    config_path = tmp_path / "spin-compact.yaml"
    config_path.write_text(
        """
base_output_dir: data/test-spin-compact
pyspi_config: configs/pyspi/test.yaml
defaults: {instances: 1, M_values: [20], T_values: [20]}
mts_classes:
  - name: mh
    generator: miller_huse
    base_params:
      lattice_side: 10
      transients: 10
      future_truth_T: 10
      store_full_field: false
""",
        encoding="utf-8",
    )
    specs = DatasetMapping(ExperimentConfig.from_file(config_path)).specs
    mh_spec = next(spec for spec in specs if spec.generator == "miller_huse")
    mh_observed, mh_extras = generate_synthetic_from_spec(mh_spec)
    assert mh_observed.shape == (20, 20)
    assert "full_field" not in mh_extras["_ground_truth"]
    assert mh_extras["_ground_truth"]["q_spin_abs"].shape == ()
    mh_semantics = _miller_huse_semantics(mh_spec, mh_extras)
    assert mh_semantics["order_parameter"]["primary_scalar"] == "q_spin_abs"


def test_desai_zwanzig_prefixes_and_future_truth_are_fixed() -> None:
    common = dict(
        M=8,
        N_full=32,
        sigma=1.8,
        burn_time=1.0,
        truth_start_T=50,
        future_truth_T=20,
        return_internals=True,
    )
    short, short_info = generate_desai_zwanzig(
        T=20, rng=np.random.default_rng(61), **common
    )
    long, long_info = generate_desai_zwanzig(
        T=50, rng=np.random.default_rng(61), **common
    )
    np.testing.assert_array_equal(short, long[:20])
    np.testing.assert_array_equal(
        short_info.mean_field_future, long_info.mean_field_future
    )
    np.testing.assert_array_equal(
        short_info.observation_indices, long_info.observation_indices
    )


def test_desai_zwanzig_reproduces_order_disorder_contrast() -> None:
    common = dict(
        M=16,
        T=200,
        N_full=512,
        burn_time=30.0,
        future_truth_T=100,
        return_internals=True,
    )
    _, ordered = generate_desai_zwanzig(
        sigma=1.5, rng=np.random.default_rng(67), **common
    )
    _, disordered = generate_desai_zwanzig(
        sigma=2.2, rng=np.random.default_rng(67), **common
    )
    assert np.mean(np.abs(ordered.mean_field_future)) > 0.5
    assert np.mean(np.abs(disordered.mean_field_future)) < 0.25


def test_desai_zwanzig_harness_records_canonical_truth(tmp_path: Path) -> None:
    config_path = tmp_path / "desai-zwanzig.yaml"
    config_path.write_text(
        """
base_output_dir: data/test-desai-zwanzig
pyspi_config: configs/pyspi/test.yaml
defaults: {instances: 1, M_values: [8], T_values: [40]}
mts_classes:
  - name: dz
    generator: desai_zwanzig
    base_params:
      N_full: 16
      sigma: 1.9
      burn_time: 1.0
      future_truth_T: 20
      store_full_states: false
""",
        encoding="utf-8",
    )
    spec = DatasetMapping(ExperimentConfig.from_file(config_path)).specs[0]
    observed, extras = generate_synthetic_from_spec(spec)
    assert observed.shape == (40, 8)
    assert "full_states" not in extras["_ground_truth"]
    assert extras["_ground_truth"]["mean_field_future"].shape == (20,)
    assert extras["_ground_truth"]["q_mean_abs_blocks"].shape == (8,)
    semantics = _desai_zwanzig_semantics(spec, extras)
    assert semantics["control"]["name"] == "sigma"
    assert semantics["order_parameter"]["primary_scalar"] == "q_mean_abs"
    assert semantics["order_parameter"]["included_in_timeseries_input"] is False


def test_stuart_landau_nested_views_and_collective_regimes() -> None:
    common = dict(
        coupling=0.8,
        N_full=64,
        dt=0.02,
        sample_dt=0.1,
        burn_time=80.0,
        return_internals=True,
    )
    short, short_info = generate_stuart_landau(
        M=8,
        T=150,
        frequency_half_width=0.6,
        rng=np.random.default_rng(41),
        **common,
    )
    long, long_info = generate_stuart_landau(
        M=16,
        T=300,
        frequency_half_width=0.6,
        rng=np.random.default_rng(41),
        **common,
    )
    np.testing.assert_allclose(short, long[:150, :8], rtol=0, atol=0)
    np.testing.assert_array_equal(
        short_info.observation_indices, long_info.observation_indices[:8]
    )
    np.testing.assert_allclose(short_info.frequencies, long_info.frequencies, rtol=0, atol=0)
    assert np.abs(short_info.order_parameter).mean() > 0.6

    _, incoherent = generate_stuart_landau(
        M=16,
        T=300,
        frequency_half_width=1.2,
        rng=np.random.default_rng(41),
        **common,
    )
    assert np.abs(incoherent.order_parameter).mean() < 0.25


def test_stuart_landau_harness_records_vector_truth(tmp_path: Path) -> None:
    config_path = tmp_path / "stuart-landau.yaml"
    config_path.write_text(
        """
base_output_dir: data/test-stuart-landau
pyspi_config: configs/pyspi/test.yaml
defaults: {instances: 1, M_values: [8], T_values: [40]}
mts_classes:
  - name: sl
    generator: stuart_landau
    base_params:
      N_full: 16
      coupling: 0.8
      frequency_half_width: 0.8
      burn_time: 5.0
      future_truth_T: 20
      store_full_states: false
""",
        encoding="utf-8",
    )
    spec = DatasetMapping(ExperimentConfig.from_file(config_path)).specs[0]
    observed, extras = generate_synthetic_from_spec(spec)
    assert observed.shape == (40, 8)
    assert "full_states_real" not in extras["_ground_truth"]
    assert extras["_ground_truth"]["order_parameter_R_future"].shape == (20,)
    semantics = _stuart_landau_semantics(spec, extras)
    assert semantics["order_parameter"]["primary_scalar"] == "q_R_mean"
    assert semantics["order_parameter"]["activity_scalar"] == "q_activity_mean"


def test_stuart_landau_prefixes_share_fixed_future_truth() -> None:
    common = dict(
        M=8,
        N_full=32,
        coupling=0.8,
        frequency_half_width=0.8,
        burn_time=20.0,
        truth_start_T=100,
        future_truth_T=40,
        return_internals=True,
    )
    short, short_info = generate_stuart_landau(
        T=30, rng=np.random.default_rng(53), **common
    )
    long, long_info = generate_stuart_landau(
        T=100, rng=np.random.default_rng(53), **common
    )
    np.testing.assert_array_equal(short, long[:30])
    np.testing.assert_array_equal(
        short_info.order_parameter_future, long_info.order_parameter_future
    )


def test_instance_seed_scope_pairs_variants_and_nested_views(tmp_path: Path) -> None:
    config_path = tmp_path / "paired.yaml"
    config_path.write_text(
        """
base_output_dir: data/test-paired
pyspi_config: configs/pyspi/test.yaml
rng_seed: 123
defaults:
  M_values: [4, 8]
  T_values: [20, 40]
  instances: 2
mts_classes:
  - name: paired
    generator: kuramoto_order_parameter
    seed_scope: instance
    include_base_variant: false
    variants:
      - {name: low, params: {K: 0.8}}
      - {name: high, params: {K: 2.4}}
""",
        encoding="utf-8",
    )
    mapping = DatasetMapping(ExperimentConfig.from_file(config_path))
    by_instance: dict[int, set[int]] = {}
    for spec in mapping.specs:
        by_instance.setdefault(spec.instance, set()).add(spec.rng_seed)
    assert all(len(seeds) == 1 for seeds in by_instance.values())
    assert next(iter(by_instance[0])) != next(iter(by_instance[1]))


def test_dataset_seed_does_not_depend_on_clone_path(tmp_path: Path) -> None:
    config = ExperimentConfig.from_file(
        Path("configs/generate/order_parameter/spin-order-smoke.yaml")
    )
    spec = DatasetMapping(config).specs[0]
    moved = replace(
        spec,
        base_output_dir=tmp_path / "another-clone" / "data",
        dataset_dir=tmp_path / "another-clone" / "data" / spec.class_dir / spec.dataset_slug,
    )
    assert _derive_dataset_seed(base_seed=config.rng_seed, spec=spec) == _derive_dataset_seed(
        base_seed=config.rng_seed, spec=moved
    )


def test_claim_benchmark_mapping_and_split_invariants() -> None:
    config = ExperimentConfig.from_file(
        Path("configs/generate/order_parameter/kuramoto-order-benchmark.yaml")
    )
    mapping = DatasetMapping(config)
    assert len(mapping.specs) == 880
    assert len({spec.dataset_dir for spec in mapping.specs}) == 880
    seen_seeds: set[int] = set()
    for class_name in (
        "kuramoto-gaussian-paired",
        "kuramoto-logistic-paired",
    ):
        paired = [spec for spec in mapping.specs if spec.mts_class == class_name]
        assert all(spec.seed_scope == "instance" for spec in paired)
        for instance in range(32):
            master = [spec for spec in paired if spec.instance == instance]
            assert len({spec.rng_seed for spec in master}) == 1
            assert len({spec.seed_group_id for spec in master}) == 1
            assert len({spec.instance < 16 for spec in master}) == 1
            seen_seeds.add(master[0].rng_seed)
    for class_name in (
        "kuramoto-gaussian-cell",
        "kuramoto-logistic-cell",
    ):
        cell = [spec for spec in mapping.specs if spec.mts_class == class_name]
        assert all(spec.seed_scope == "dataset" for spec in cell)
        assert len({spec.rng_seed for spec in cell}) == len(cell)
        assert len({spec.seed_group_id for spec in cell}) == len(cell)
        seen_seeds.update(spec.rng_seed for spec in cell)
    assert len(seen_seeds) == 64 + 80 + 96
    assert all(spec.generator_params["future_truth_T"] == 1000 for spec in mapping.specs)
    assert all(spec.generator_params["store_full_phases"] is False for spec in mapping.specs)


def test_kuramoto_confirmation_uses_new_controls_and_both_sampling_designs() -> None:
    old = DatasetMapping(
        ExperimentConfig.from_file(
            Path("configs/generate/order_parameter/kuramoto-order-benchmark.yaml")
        )
    )
    confirmation = DatasetMapping(
        ExperimentConfig.from_file(
            Path("configs/generate/order_parameter/kuramoto-confirmation.yaml")
        )
    )
    assert len(confirmation.specs) == 1152
    assert {spec.M for spec in confirmation.specs} == {20}
    assert {spec.T for spec in confirmation.specs} == {1000}
    old_controls = {
        round(float(spec.generator_params["K"]), 10) for spec in old.specs
    }
    new_controls = {
        round(float(spec.generator_params["K"]), 10) for spec in confirmation.specs
    }
    assert old_controls.isdisjoint(new_controls)
    assert {spec.generator_params["frequency_sampling"] for spec in confirmation.specs} == {
        "random",
        "regular",
    }
    assert all(
        "paired" in spec.class_labels
        for spec in confirmation.specs
        if spec.generator_params["frequency_sampling"] == "regular"
    )
    assert all(spec.generator_params["future_truth_T"] == 1000 for spec in confirmation.specs)
    assert all(spec.generator_params["store_full_phases"] is False for spec in confirmation.specs)


def test_terminal_kuramoto_confirmation_is_independent_and_preserves_hard_paths() -> None:
    mappings = [
        DatasetMapping(ExperimentConfig.from_file(Path(path)))
        for path in (
            "configs/generate/order_parameter/kuramoto-order-benchmark.yaml",
            "configs/generate/order_parameter/kuramoto-confirmation.yaml",
            "configs/generate/order_parameter/kuramoto-final-confirmation.yaml",
        )
    ]
    final = mappings[2]
    assert len(final.specs) == 1536
    assert {spec.M for spec in final.specs} == {20}
    assert {spec.T for spec in final.specs} == {1000}
    controls = [
        {round(float(spec.generator_params["K"]), 10) for spec in mapping.specs}
        for mapping in mappings
    ]
    assert controls[2].isdisjoint(controls[0])
    assert controls[2].isdisjoint(controls[1])
    assert {
        (
            spec.generator_params["frequency_distribution"],
            spec.generator_params["frequency_sampling"],
            spec.seed_scope,
        )
        for spec in final.specs
    } == {
        ("gaussian", "random", "instance"),
        ("gaussian", "random", "dataset"),
        ("gaussian", "regular", "instance"),
        ("logistic", "random", "instance"),
        ("logistic", "random", "dataset"),
        ("logistic", "regular", "instance"),
    }
    assert all(spec.generator_params["future_truth_T"] == 1000 for spec in final.specs)
    assert all(spec.generator_params["store_full_phases"] is False for spec in final.specs)


def test_explicit_instance_indices_are_preserved(tmp_path: Path) -> None:
    config_path = tmp_path / "explicit-instances.yaml"
    config_path.write_text(
        """
base_output_dir: data/test-explicit-instances
timestamp: false
pyspi_config: configs/pyspi/benchmarked_p90.yaml
normalise: false
rng_seed: 7
save_heatmap: false
defaults:
  M_values: [8]
  T_values: [500]
  instances: [10, 12, 29]
mts_classes:
  - name: gaussian-noise
    generator: gaussian_noise
    base_params: {zscore: false}
""",
        encoding="utf-8",
    )

    mapping = DatasetMapping(ExperimentConfig.from_file(config_path))

    assert [spec.instance for spec in mapping.specs] == [10, 12, 29]
