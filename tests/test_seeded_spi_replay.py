import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.compute import run_pyspi, seeded_estimator_rng
from scripts.replay_proof_sgddtw import CONFIG, NAMES, replay


def test_seeded_sgd_is_repeatable_and_rng_is_restored():
    data = np.random.default_rng(14).normal(size=(30, 3))
    state = np.random.get_state()
    python_state = random.getstate()
    first = run_pyspi(data, config_path=CONFIG, n_jobs=1, random_seed=17)
    second = run_pyspi(data, config_path=CONFIG, n_jobs=1, random_seed=17)
    assert set(first.matrices) == set(NAMES)
    for name in NAMES:
        np.testing.assert_array_equal(first.matrices[name], second.matrices[name])
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]
    assert random.getstate() == python_state
    with pytest.raises(RuntimeError), seeded_estimator_rng(18):
        np.random.random()
        raise RuntimeError()
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])


def test_partial_replay_preserves_other_values_and_resumes(tmp_path):
    original = {name: np.eye(3) for name in NAMES}
    original["untouched"] = np.array([[np.nan, 1.], [2., 0.]])
    np.savez_compressed(tmp_path / "spi_mpis.npz", **original)
    np.save(tmp_path / "timeseries.npy", np.ones((20, 3)))
    meta_path = tmp_path / "meta.json"
    metadata = {"generator": {"seed": 19}, "normalise": False,
                "pyspi": {"errors": {"untouched": "original", NAMES[0]: "old"}}}
    meta_path.write_text(json.dumps(metadata))
    calls = []
    def fake(data, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(matrices={name: np.full((3, 3), 2.) for name in NAMES},
                               errors={}, timings={})
    replay(tmp_path, compute=fake)
    # Simulate a crash after the MPI replacement but before metadata replacement.
    meta_path.write_text(json.dumps(metadata))
    replay(tmp_path, compute=fake)
    assert len(calls) == 1 and calls[0]["random_seed"] == 19
    with np.load(tmp_path / "spi_mpis.npz") as archive:
        np.testing.assert_array_equal(archive["untouched"], original["untouched"])
        for name in NAMES:
            np.testing.assert_array_equal(archive[name], np.full((3, 3), 2.))
    assert json.loads(meta_path.read_text())["pyspi"]["errors"] == {"untouched": "original"}
    with np.load(tmp_path / "sgddtw-seeded-replay.npz") as archive:
        np.testing.assert_array_equal(archive[f"original_{NAMES[0]}"], original[NAMES[0]])
