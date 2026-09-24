"""Fresh and resumed extraction must receive identical synthetic inputs."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.mapping import DatasetMapping, ExperimentConfig
from src.run_experiments import _ensure_timeseries, generate_synthetic_from_spec


@pytest.mark.parametrize("config", [
    "embeddings/proof-p90-260924.yaml",
    "r_rho_mi/260924_beta.yaml",
    "dtw_euclidean/260924_lagged-warping.yaml",
])
def test_saved_input_equals_fresh_and_resumed_input(tmp_path, config):
    spec = DatasetMapping(ExperimentConfig.from_file(
        Path(__file__).resolve().parents[1] / "configs/generate" / config
    )).spec_for_index(1)
    spec = replace(spec, dataset_dir=tmp_path)
    expected, extras = generate_synthetic_from_spec(spec)
    fresh, path, _ = _ensure_timeseries(spec, regenerate=False)
    resumed, _, _ = _ensure_timeseries(spec, regenerate=False)
    saved = np.load(path)
    assert saved.dtype == np.float64
    np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(fresh, resumed)
    if "_mother" in extras:
        np.testing.assert_array_equal(np.load(tmp_path / "mother.npy"), extras["_mother"])
