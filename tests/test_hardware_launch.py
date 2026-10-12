import pytest

from scripts.launch_hardware_sweep import resource_plan


@pytest.mark.parametrize('peak', [1, 5.31, 8, 12, 14])
def test_each_node_covers_its_concurrent_workers(peak):
    plan = resource_plan(peak, 4 * 3600)
    workers_per_node = (plan['workers'] + plan['nodes'] - 1) // plan['nodes']
    assert plan['mem_gb'] / plan['nodes'] >= workers_per_node * plan['per_worker_gib']
    assert plan['mem_gb'] / plan['nodes'] <= 180
    assert plan['per_worker_gib'] >= 1.5 * peak
    assert plan['ncpus'] >= plan['workers']


def test_unexpected_memory_blocks_automatic_scaling():
    with pytest.raises(ValueError, match='bounded pilot'):
        resource_plan(40, 4 * 3600)


@pytest.mark.parametrize('smallest', [2500, 3125, 4000, None])
def test_smaller_candidate_or_failed_sampling_holds_release(tmp_path, smallest):
    import json
    from scripts.launch_hardware_sweep import validate_sampling
    path = tmp_path / 'scope-er-intermediate'
    path.mkdir()
    (path / 'sampling-decision.json').write_text(json.dumps(
        dict(records=29, smallest_passing_T=smallest)))
    with pytest.raises(ValueError, match='matched SPI structure'):
        validate_sampling(tmp_path)
