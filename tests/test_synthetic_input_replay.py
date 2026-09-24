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


@pytest.mark.parametrize("base,extra", [
    ("r_rho_mi/260924_beta.yaml", "r_rho_mi/260924_beta-extra.yaml"),
    ("dtw_euclidean/260924_lagged-warping.yaml", "dtw_euclidean/260924_lagged-warping-extra.yaml"),
])
def test_case_extensions_are_disjoint_and_cover_100_instances(base, extra):
    root = Path(__file__).resolve().parents[1] / "configs/generate"
    specs = [s for name in (base, extra) for s in DatasetMapping(ExperimentConfig.from_file(root/name)).specs]
    assert len(specs) == len({s.dataset_dir for s in specs}) == 700
    conditions = {}
    for spec in specs:
        key = (spec.mts_class, spec.variant.slug if spec.variant else "")
        conditions.setdefault(key, []).append(spec.instance)
    assert len(conditions) == 7
    assert all(sorted(instances) == list(range(100)) for instances in conditions.values())


def test_proof_var_contrasts_are_literal_and_stable():
    root = Path(__file__).resolve().parents[1]
    specs = DatasetMapping(ExperimentConfig.from_file(root/"configs/generate/embeddings/proof-p90-260924.yaml")).specs
    params = {}
    for spec in specs:
        if spec.generator != "varma":
            continue
        p = spec.generator_params
        params[spec.mts_class] = (p["phi"], 2*p["coupling"])
        identity = np.eye(spec.M)
        matrix = p["phi"]*identity + p["coupling"]*(np.roll(identity,1,0)+np.roll(identity,-1,0))
        assert max(abs(np.linalg.eigvalsh(matrix))) < .98
    assert set(params.values()) == {(.7,.2),(.2,.2),(.2,.7)}
