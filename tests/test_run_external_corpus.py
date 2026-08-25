import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.compute import ComputeResult, SPIInfo
from src.run_external_corpus import (
    ExternalCorpusConfig,
    completion_error,
    load_inventory,
    run_dataset,
    validate_source,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    archive_path = tmp_path / "source.npz"
    arrays = {
        "first dataset": np.arange(15, dtype=np.float64).reshape(3, 5),
        "second": np.arange(16, dtype=np.int64).reshape(2, 8),
        "__dataset_names__": np.asarray(["first dataset", "second"]),
        "__labels_json__": np.asarray(['["synthetic","var"]', '["real"]']),
        "__shapes__": np.asarray([[3, 5], [2, 8]], dtype=np.int32),
        "__axis_order__": np.asarray(["process", "observation"]),
    }
    np.savez_compressed(archive_path, **arrays)
    pyspi_config = tmp_path / "pyspi.yaml"
    pyspi_config.write_text("dummy: true\n", encoding="utf-8")
    config_path = tmp_path / "corpus.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: fixture",
                "source:",
                "  format: named-npz-v1",
                f"  archive: {archive_path}",
                f"  sha256: {_sha256(archive_path)}",
                "  axis_order: [process, observation]",
                f"base_output_dir: {tmp_path / 'outputs'}",
                f"pyspi_config: {pyspi_config}",
                "normalise: true",
                "random_seed: 1729",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path, archive_path


def test_inventory_and_source_validation(tmp_path: Path) -> None:
    config_path, archive_path = _fixture(tmp_path)
    config = ExternalCorpusConfig.from_file(config_path)
    entries = load_inventory(config)
    assert [(entry.index, entry.M, entry.T) for entry in entries] == [(1, 3, 5), (2, 2, 8)]
    assert entries[0].labels == ("synthetic", "var")
    summary = validate_source(config)
    assert summary["datasets"] == 2
    assert summary["sha256"] == _sha256(archive_path)


def test_run_transposes_source_and_writes_compatible_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = _fixture(tmp_path)
    config = ExternalCorpusConfig.from_file(config_path)
    entry = load_inventory(config)[0]
    observed: dict[str, object] = {}

    def fake_run_pyspi(data, *, config_path, normalise, n_jobs):
        observed["shape"] = data.shape
        observed["normalise"] = normalise
        observed["n_jobs"] = n_jobs
        spi = SPIInfo(
            name="pearson", directed=False, labels=["undirected"],
            family="basic", module="pyspi.statistics.basic", class_name="PearsonR",
        )
        return ComputeResult(
            table=None,  # type: ignore[arg-type]
            matrices={"pearson": np.eye(3)},
            metadata=[spi],
            timings={"pearson": 0.01},
            errors={},
        )

    monkeypatch.setattr("src.run_external_corpus.run_pyspi", fake_run_pyspi)
    output = run_dataset(config, entry, n_jobs=1)
    assert observed == {"shape": (5, 3), "normalise": True, "n_jobs": 1}
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert meta["mts_class"] == "first dataset"
    assert meta["random_seed"] == 1729
    assert meta["labels"] == ["synthetic", "var"]
    assert meta["source"]["source_shape"] == [3, 5]
    assert meta["pyspi"]["spis"][0]["name"] == "pearson"
    assert completion_error(config, entry) is None

    with np.load(output / "spi_mpis.npz", allow_pickle=False) as archive:
        assert archive.files == ["pearson"]
        assert archive["pearson"].shape == (3, 3)


def test_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    config_path, archive_path = _fixture(tmp_path)
    config = ExternalCorpusConfig.from_file(config_path)
    with archive_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_source(config)


def test_runner_pins_and_restores_global_random_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = _fixture(tmp_path)
    config = ExternalCorpusConfig.from_file(config_path)
    entry = load_inventory(config)[0]
    observed: list[tuple[float, float]] = []

    def fake_run_pyspi(data, *, config_path, normalise, n_jobs):
        observed.append((float(np.random.random()), random.random()))
        spi = SPIInfo(
            name="pearson",
            directed=False,
            labels=["undirected"],
            family="basic",
            module="pyspi.statistics.basic",
            class_name="PearsonR",
        )
        return ComputeResult(
            table=None,  # type: ignore[arg-type]
            matrices={"pearson": np.eye(3)},
            metadata=[spi],
            timings={},
            errors={},
        )

    import random

    monkeypatch.setattr("src.run_external_corpus.run_pyspi", fake_run_pyspi)
    np.random.seed(23)
    random.seed(23)
    expected_numpy = np.random.random()
    expected_python = random.random()
    np.random.seed(23)
    random.seed(23)
    run_dataset(config, entry)
    after_numpy = np.random.random()
    after_python = random.random()
    run_dataset(config, entry)

    np.random.seed(1729)
    random.seed(1729)
    assert observed[0] == (float(np.random.random()), random.random())
    assert observed[0] == observed[1]
    assert after_numpy == expected_numpy
    assert after_python == expected_python
