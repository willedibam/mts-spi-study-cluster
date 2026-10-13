"""Fresh and resumed extraction must receive identical synthetic inputs."""
from pathlib import Path

import numpy as np
import pytest

from mtsspi.mapping import DatasetMapping, ExperimentConfig
from mtsspi.generators.harness import generate_from_spec
from mtsspi.generated_corpus import GeneratedCorpus


@pytest.mark.parametrize("config", [
    "configs/generate/embeddings/proof-p90-260924.yaml",
    "research/cases/r_rho_mi/configs/260924_beta.yaml",
    "research/cases/dtw_euclidean/configs/260924_lagged-warping.yaml",
])
def test_saved_input_equals_fresh_and_resumed_input(tmp_path, config):
    corpus = GeneratedCorpus.from_file(
        Path(__file__).resolve().parents[1] / config)
    entry = corpus.entries()[0]
    spec = entry.spec
    spec.dataset_dir = tmp_path
    expected, extras = generate_from_spec(spec)
    fresh, _ = corpus.load(entry)
    resumed, _ = corpus.load(entry)
    path = tmp_path / "timeseries.npy"
    saved = np.load(path)
    assert saved.dtype == np.float64
    np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(fresh, resumed)
    if "_mother" in extras:
        np.testing.assert_array_equal(np.load(tmp_path / "mother.npy"), extras["_mother"])


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
