from pathlib import Path

import numpy as np

from src.generators.order_parameter import (
    generate_kuramoto_order_parameter,
    generate_miller_huse,
    kuramoto_critical_coupling,
    miller_huse_map,
)
from src.mapping import DatasetMapping, ExperimentConfig


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
    observed, info = generate_kuramoto_order_parameter(
        M=4,
        T=20,
        future_truth_T=15,
        N_full=24,
        burn_time=2.0,
        rng=np.random.default_rng(19),
        return_internals=True,
    )
    assert observed.shape == (20, 4)
    assert info.full_phases.shape == (20, 24)
    assert info.r_full.shape == (20,)
    assert info.r_full_future.shape == (15,)
    assert info.r_unobserved_future.shape == (15,)


def test_miller_huse_map_and_patch_shapes() -> None:
    values = np.linspace(-1.0, 1.0, 101)
    mapped = miller_huse_map(values)
    np.testing.assert_allclose(miller_huse_map(-values), -mapped, atol=1e-14)
    assert np.max(np.abs(mapped)) <= 1.0 + 1e-14

    observed, internals = generate_miller_huse(
        M=25,
        T=30,
        coupling=0.2,
        lattice_side=16,
        transients=100,
        rng=np.random.default_rng(3),
        return_internals=True,
    )
    assert observed.shape == (30, 25)
    assert internals.full_field.shape == (30, 16, 16)
    assert internals.patch_indices.shape == (25, 2)
    assert np.max(np.abs(internals.full_field)) <= 1.0 + 1e-12
    assert np.all(np.abs(internals.spin_magnetization) <= 1.0)


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
