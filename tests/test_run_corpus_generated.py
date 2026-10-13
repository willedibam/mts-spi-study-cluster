"""A generated corpus runs under the same contract as an archive corpus."""
import json

import numpy as np
import pytest

from mtsspi import provenance, run_corpus
from mtsspi.compute import ComputeResult, SPIInfo

CONFIG = """
base_output_dir: {out}
pyspi_config: {spis}
normalise: false
rng_seed: 7
defaults: {{M_values: [3], T_values: [40], instances: 2}}
mts_classes:
  - name: Noise
    labels: [noise]
    generator: gaussian_noise
    base_params: {{zscore: true}}
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    spis = tmp_path / "spis.yaml"
    spis.write_text(".statistics.basic:\n  Covariance:\n    labels: [undirected]\n")
    config = tmp_path / "corpus.yaml"
    config.write_text(CONFIG.format(out=tmp_path / "out", spis=spis))

    def fake_run_pyspi(data, *, config_path, normalise, n_jobs, random_seed, **options):
        assert data.shape == (40, 3) and data.dtype == np.float64
        spi = SPIInfo(name="cov", directed=False, labels=["undirected"], family="basic",
                      module="pyspi.statistics.basic", class_name="Covariance")
        return ComputeResult(table=None, matrices={"cov": np.eye(3)}, metadata=[spi],
                             timings={"cov": 0.01}, errors={})

    monkeypatch.setattr(run_corpus, "run_pyspi", fake_run_pyspi)
    monkeypatch.setattr(provenance, "bound_cores", lambda: 4)
    for name in provenance.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    return run_corpus.load_corpus(config)


def test_generated_recording_is_written_validated_and_skipped(corpus, capsys):
    first, second = corpus.entries()
    assert (first.name, first.M, first.T) == ("Noise_M3_T40_I0", 3, 40)
    assert run_corpus.completion_error(corpus, first) == "missing meta.json or spi_mpis.npz"
    directory = run_corpus.run_dataset(corpus, first, n_jobs=2)
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["mts_class"] == "Noise"
    assert meta["random_seed"] == first.spec.rng_seed == meta["generator"]["seed"]
    assert meta["job"]["n_jobs"] == 2 and meta["job"]["estimator_rng_policy"] == "pyspi_fixed_seeds"
    assert np.load(directory / "timeseries.npy").shape == (40, 3)
    assert run_corpus.completion_error(corpus, first) is None
    assert run_corpus.completion_error(corpus, second) is not None
    run_corpus.run_dataset(corpus, first, skip_existing=True)
    assert "[SKIP]" in capsys.readouterr().out


def test_result_from_another_pyspi_revision_is_not_complete(corpus):
    entry = corpus.entries()[0]
    directory = run_corpus.run_dataset(corpus, entry)
    path = directory / "meta.json"
    meta = json.loads(path.read_text())
    meta["execution_identity"]["pyspi_version"]["computation"] = "3.0.0.r7"
    path.write_text(json.dumps(meta))
    assert run_corpus.completion_error(corpus, entry) == "execution identity mismatch: pyspi_version"
    audit = run_corpus.audit_outputs(corpus, [entry])
    assert audit["complete"] == 0 and audit["failures"][0]["index"] == 1


def test_a_lost_series_makes_the_recording_incomplete(corpus):
    entry = corpus.entries()[0]
    directory = run_corpus.run_dataset(corpus, entry)
    (directory / "timeseries.npy").unlink()
    assert run_corpus.completion_error(corpus, entry) == "missing timeseries.npy"


def test_real_data_classes_are_refused_with_the_way_forward(tmp_path):
    config = tmp_path / "real.yaml"
    config.write_text("pyspi_config: x.yaml\nmts_classes:\n  - name: UEA\n    package: aeon\n    dataset_name: Epilepsy\n")
    with pytest.raises(ValueError, match="prepare them as an archive"):
        run_corpus.load_corpus(config)


def test_cli_counts_and_lists_a_generated_corpus(corpus, capsys):
    run_corpus.main(["--config", str(corpus.path), "--count-only"])
    assert capsys.readouterr().out.strip() == "2"
    run_corpus.main(["--config", str(corpus.path), "--list"])
    assert capsys.readouterr().out.splitlines()[1].split("\t") == ["2", "3", "40", "Noise_M3_T40_I1"]


def test_a_config_that_expands_to_nothing_is_an_error(tmp_path):
    config = tmp_path / "empty.yaml"
    config.write_text("pyspi_config: x.yaml\nmts_classes:\n  - name: N\n    generator: gaussian_noise\n    M_values: [3]\n    T_values: [40]\n")
    with pytest.raises(ValueError, match="expands to no datasets"):
        run_corpus.load_corpus(config)

