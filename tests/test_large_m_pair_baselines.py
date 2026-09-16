import numpy as np
import pytest
from scripts.large_m_pair_baselines import assemble, standardize, tasks, write_part
from scripts.prepare_hcp_m32 import farthest_sensors


def test_plan_has_expected_families_sizes_and_counts():
    plan = tasks()
    assert len(plan) == 25
    assert sum(t['system'] == 'cml' for t in plan) == 6
    assert sum(t['system'] == 'tasep' for t in plan) == 18
    assert plan[-1] == dict(system='meg')
    for m in [64, 100, 256]:
        assert sum(t.get('M') == m for t in plan) == 6


def test_standardization_is_nested_and_does_not_modify_input():
    original = np.random.default_rng(5).normal(size=(1000, 256))
    copy = original.copy()
    full = standardize(original)
    np.testing.assert_allclose(standardize(original[:, :100]), full[:, :100])
    np.testing.assert_array_equal(original, copy)
    np.testing.assert_allclose(full.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(full.std(axis=0), 1)
    with pytest.raises(AssertionError): standardize(np.ones((1000, 3)))


def test_full_sensor_order_is_unique_and_preserves_coverage_prefix():
    positions = np.random.default_rng(5).normal(size=(243, 3))
    full, _ = farthest_sensors(positions, 0, count=243)
    prefix, _ = farthest_sensors(positions, 0, count=64)
    assert len(set(full)) == 243
    np.testing.assert_array_equal(full[:64], prefix)


def test_assembly_produces_three_valid_16_record_groups(tmp_path):
    from src.run_external_corpus import ExternalCorpusConfig, load_inventory
    for index, task in enumerate(tasks()):
        if task['system'] == 'cml':
            sizes = [64, 100, 256]
        elif task['system'] == 'tasep':
            sizes = [task['M']]
        else:
            sizes = [64, 100, 243] * 4
        arrays, rows = {}, []
        for j, m in enumerate(sizes):
            name = f'fixture-{index}-{j}'
            arrays[name] = np.broadcast_to(np.arange(1000)[:, None], (1000, m))
            rows.append(dict(name=name, M=m, T=1000, system=task['system']))
        write_part(tmp_path, index, arrays, rows, {})
    assemble(tmp_path)
    for group in ['m64', 'm100', 'large']:
        config = ExternalCorpusConfig.from_file(tmp_path/f'{group}.yaml')
        assert len(load_inventory(config)) == 16
        smoke = (tmp_path/f'{group}-smoke.txt').read_text().split()
        rest = (tmp_path/f'{group}-rest.txt').read_text().split()
        assert len(smoke) == 3 and len(rest) == 13
        assert not set(smoke).intersection(rest)
