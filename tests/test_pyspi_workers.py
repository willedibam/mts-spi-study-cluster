"""Several pyspi workers per recording: what the wrappers require and record."""
import hashlib
import json

import numpy as np
import pytest

from src import compute, run_external_corpus as corpus
from src.compute import pyspi_pins_estimator_seeds, run_pyspi

STOCHASTIC_CONFIG = """
.statistics.distance:
  Barycenter:
    configs:
      - mode: sgddtw
        statistic: mean
.statistics.basic:
  Covariance:
    configs:
      - estimator: MinCovDet
  SpearmanR:
    configs:
      - squared: False
"""


def _series(seed=0, T=120, M=4):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, M))
    for t in range(1, T):
        x[t] += 0.6 * x[t - 1]
    return x


def _corpus(tmp_path):
    archive = tmp_path / "source.npz"
    np.savez_compressed(
        archive,
        **{"only": np.arange(15, dtype=np.float64).reshape(3, 5),
           "__dataset_names__": np.asarray(["only"]),
           "__labels_json__": np.asarray(['["synthetic"]']),
           "__shapes__": np.asarray([[3, 5]], dtype=np.int32),
           "__axis_order__": np.asarray(["process", "observation"])})
    pyspi_config = tmp_path / "pyspi.yaml"
    pyspi_config.write_text("dummy: true\n", encoding="utf-8")
    path = tmp_path / "corpus.yaml"
    path.write_text("\n".join([
        "name: fixture", "source:", "  format: named-npz-v1",
        f"  archive: {archive}",
        f"  sha256: {hashlib.sha256(archive.read_bytes()).hexdigest()}",
        "  axis_order: [process, observation]",
        f"base_output_dir: {tmp_path / 'outputs'}",
        f"pyspi_config: {pyspi_config}", "normalise: true", "random_seed: 1729", ""]),
        encoding="utf-8")
    return corpus.ExternalCorpusConfig.from_file(path)


@pytest.fixture
def stochastic_config(tmp_path):
    path = tmp_path / "stochastic.yaml"
    path.write_text(STOCHASTIC_CONFIG, encoding="utf-8")
    return path


def test_unpinned_pyspi_still_needs_one_worker_to_seed(monkeypatch, stochastic_config):
    monkeypatch.setattr(compute, "pyspi_pins_estimator_seeds", lambda: False)
    with pytest.raises(ValueError, match="n_jobs=1"):
        run_pyspi(_series(), config_path=stochastic_config, n_jobs=2, random_seed=7)


@pytest.mark.skipif(not pyspi_pins_estimator_seeds(),
                    reason="this pyspi draws estimator randomness from the global stream")
def test_pinned_pyspi_gives_one_result_for_any_worker_count_and_seed(stochastic_config):
    """Stochastic estimators included: values, missing entries and errors."""
    runs = [run_pyspi(_series(), config_path=stochastic_config, n_jobs=n_jobs,
                      random_seed=seed, mp_context=context)
            for n_jobs, seed, context in ((1, 7, None), (1, 8, None), (2, 7, "spawn"))]
    first = runs[0]
    assert first.rng_policy == "pyspi_fixed_seeds"
    assert len(first.matrices) == 3
    for other in runs[1:]:
        assert list(other.matrices) == list(first.matrices)
        assert other.errors == first.errors
        for name, matrix in first.matrices.items():
            assert np.array_equal(other.matrices[name], matrix, equal_nan=True), name


def test_runner_refuses_more_workers_than_bound_cores(monkeypatch):
    monkeypatch.setattr(corpus, "bound_cores", lambda: 2)
    for name in corpus._THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    assert corpus.worker_resources(2)["bound_cores"] == 2
    with pytest.raises(RuntimeError, match="bound to"):
        corpus.worker_resources(3)


def test_runner_refuses_threaded_libraries_under_several_workers(monkeypatch):
    monkeypatch.setattr(corpus, "bound_cores", lambda: 8)
    for name in corpus._THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS"):
        corpus.worker_resources(2)
    monkeypatch.delenv("MKL_NUM_THREADS")
    with pytest.raises(RuntimeError, match="MKL_NUM_THREADS"):
        corpus.worker_resources(4)
    # One worker keeps the previous behaviour: nothing is demanded of it.
    assert corpus.worker_resources(1)["thread_variables"]["OMP_NUM_THREADS"] == "4"


@pytest.mark.parametrize("pinned,n_jobs,seeded", [
    (True, 1, True), (True, 4, True), (False, 1, True), (False, 4, False)])
def test_runner_seeds_whenever_the_result_cannot_depend_on_workers(
        tmp_path, monkeypatch, pinned, n_jobs, seeded):
    from src.compute import ComputeResult, SPIInfo

    config = _corpus(tmp_path)
    entry = corpus.load_inventory(config)[0]
    seen = {}

    def fake_run_pyspi(data, *, config_path, normalise, n_jobs, random_seed):
        seen["random_seed"] = random_seed
        spi = SPIInfo(name="pearson", directed=False, labels=["undirected"],
                      family="basic", module="pyspi.statistics.basic", class_name="PearsonR")
        return ComputeResult(table=None, matrices={"pearson": np.eye(3)}, metadata=[spi],
                             timings={"pearson": 0.01}, errors={},
                             rng_policy="pyspi_fixed_seeds" if pinned else None)

    monkeypatch.setattr(corpus, "run_pyspi", fake_run_pyspi)
    monkeypatch.setattr(corpus, "pyspi_pins_estimator_seeds", lambda: pinned)
    monkeypatch.setattr(corpus, "bound_cores", lambda: 8)
    for name in corpus._THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")

    output = corpus.run_dataset(config, entry, n_jobs=n_jobs)
    assert seen["random_seed"] == (config.random_seed if seeded else None)
    job = json.loads((output / "meta.json").read_text(encoding="utf-8"))["job"]
    assert job["n_jobs"] == n_jobs and job["bound_cores"] == 8
    assert job["estimator_rng_policy"] == (
        "pyspi_fixed_seeds" if pinned else
        "isolated_serial" if n_jobs == 1 else "parallel_uncontrolled")
    assert job["peak_memory_mib"]["parent"] > 0
    assert corpus.completion_error(config, entry) is None


def _real_corpus(tmp_path, pair_tolerance=None):
    series = _series(T=150, M=6)
    archive = tmp_path / "source.npz"
    np.savez_compressed(
        archive,
        **{"only": series.T,
           "__dataset_names__": np.asarray(["only"]),
           "__labels_json__": np.asarray(['["synthetic"]']),
           "__shapes__": np.asarray([[6, 150]], dtype=np.int32),
           "__axis_order__": np.asarray(["process", "observation"])})
    pyspi_config = tmp_path / "pyspi.yaml"
    pyspi_config.write_text(STOCHASTIC_CONFIG, encoding="utf-8")
    path = tmp_path / "corpus.yaml"
    path.write_text("\n".join([
        "name: fixture", "source:", "  format: named-npz-v1",
        f"  archive: {archive}",
        f"  sha256: {hashlib.sha256(archive.read_bytes()).hexdigest()}",
        "  axis_order: [process, observation]",
        f"base_output_dir: {tmp_path / ('sampled' if pair_tolerance else 'full')}",
        f"pyspi_config: {pyspi_config}", "normalise: false", "random_seed: 1729",
        *([f"pair_tolerance: {pair_tolerance}"] if pair_tolerance else []), ""]),
        encoding="utf-8")
    return corpus.ExternalCorpusConfig.from_file(path)


@pytest.mark.skipif(not hasattr(compute.Calculator, "compute_pairs"),
                    reason="this pyspi has no dyad sampling")
def test_sampled_corpus_run_holds_the_full_run_at_its_dyads(tmp_path):
    """Same archive layout; unsampled entries are NaN and the dyads are stored."""
    outputs = {}
    for label, tolerance in (("full", None), ("sampled", 0.5)):
        config = _real_corpus(tmp_path, tolerance)
        entry = corpus.load_inventory(config)[0]
        directory = corpus.run_dataset(config, entry, n_jobs=1)
        assert corpus.completion_error(config, entry) is None
        with np.load(directory / "spi_mpis.npz") as archive:
            outputs[label] = (directory, {name: archive[name] for name in archive.files})
    directory, sampled = outputs["sampled"]
    full = outputs["full"][1]
    pairs = np.load(directory / "pairs.npz")["pairs"]
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["pyspi"]["sampling"]["tolerance"] == 0.5
    assert len(pairs) == meta["pyspi"]["sampling"]["n_pairs"] < 15
    held = np.eye(6, dtype=bool)
    held[pairs[:, 0], pairs[:, 1]] = held[pairs[:, 1], pairs[:, 0]] = True
    assert list(sampled) == list(full)
    for name, matrix in sampled.items():
        assert np.array_equal(matrix[held], full[name][held]), name
        assert np.isnan(matrix[~held]).all(), name

    # A full-table config must not accept the sampled output, nor the reverse.
    assert corpus.completion_error(
        _real_corpus(tmp_path, 0.4), corpus.load_inventory(_real_corpus(tmp_path, 0.4))[0]
    ) is not None


def test_pair_tolerance_must_be_a_fraction(tmp_path):
    config = _corpus(tmp_path)
    text = config.path.read_text(encoding="utf-8")
    for bad in ("3", "0.0", "true", "1.5"):
        config.path.write_text(text + f"pair_tolerance: {bad}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pair_tolerance"):
            corpus.ExternalCorpusConfig.from_file(config.path)
