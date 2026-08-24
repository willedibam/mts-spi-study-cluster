import json
from pathlib import Path

import numpy as np
import pytest

from src.process_features import main


SPI_CATALOG = [
    {"name": "u1", "directed": False},
    {"name": "d", "directed": True},
    {"name": "u2", "directed": False},
]


def _write_sample(path: Path, instance: int, *, constant_directed: bool) -> None:
    path.mkdir(parents=True)
    metadata = {
        "mts_class": "contract-test",
        "labels": ["test"],
        "M": 3,
        "T": 20,
        "instance_index": instance,
        "normalise": False,
        "experiment": {
            "config_sha256": "experiment-sha",
            "git_commit": "generation-commit",
        },
        "pyspi": {
            "config": "configs/pyspi/benchmarked_p90.yaml",
            "config_sha256": "pyspi-config-sha",
            "version": {"dist": "3.0.0", "computation": "3.0.0.r7"},
            "spis": SPI_CATALOG,
        },
    }
    (path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    u1 = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 4.0], [2.0, 4.0, 0.0]])
    u2 = np.array([[0.0, 2.0, 5.0], [2.0, 0.0, 1.0], [5.0, 1.0, 0.0]])
    directed = (
        np.ones((3, 3), dtype=float)
        if constant_directed
        else np.array([[0.0, 1.0, 7.0], [4.0, 0.0, 2.0], [3.0, 6.0, 0.0]])
    )
    np.savez(path / "spi_mpis.npz", u1=u1, d=directed, u2=u2)


def test_v2_artifact_has_frozen_complete_schema_and_validated_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_sample(data_root / "contract-test" / "sample-0", 0, constant_directed=False)
    _write_sample(data_root / "contract-test" / "sample-1", 1, constant_directed=True)
    output = tmp_path / "features.npz"

    arguments = [
        "--data-path",
        str(data_root),
        "--metric",
        "pearson",
        "--feature-contract",
        "direction_preserving_v2",
        "--output",
        str(output),
        "--workers",
        "2",
    ]
    main(arguments)
    with np.load(output, allow_pickle=True) as archive:
        assert archive["feature_contract"].item() == "direction_preserving_v2"
        assert archive["X_sym"].shape == (2, 3)
        assert archive["X_dir"].shape == (2, 3)
        assert archive["sym_validity_mask"].shape == (2, 3)
        assert archive["dir_validity_mask"].shape == (2, 3)
        assert np.isnan(archive["X_sym"][1]).any()
        assert (~archive["dir_validity_mask"][1]).any()
        assert archive["feature_block"].tolist() == [
            "sym",
            "sym",
            "sym",
            "dir",
            "dir",
            "dir",
        ]
        provenance = json.loads(archive["pyspi_provenance_json"].item())
        assert provenance["status"] == "complete"

    # An unchanged source and builder must validate and reuse the artifact.
    main(arguments)

    # Source content, not just its path, participates in cache identity.
    sample_path = data_root / "contract-test" / "sample-0" / "spi_mpis.npz"
    with np.load(sample_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["d"] = arrays["d"].copy()
    arrays["d"][0, 1] += 0.25
    np.savez(sample_path, **arrays)
    with pytest.raises(ValueError, match="cache identity mismatch"):
        main(arguments)


def test_v2_rejects_corpus_variance_filter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never applies a corpus variance filter"):
        main(
            [
                "--data-path",
                str(tmp_path),
                "--feature-contract",
                "direction_preserving_v2",
                "--var-threshold",
                "1e-8",
            ]
        )


def test_kuramoto_subset_matches_frozen_final_core() -> None:
    root = Path(__file__).resolve().parents[1]
    subset = (
        root / "configs/pyspi/subsets/kuramoto-final-core.txt"
    ).read_text(encoding="utf-8").splitlines()
    with np.load(
        root
        / "data/order_parameter/kuramoto_final_confirmation_contract/representation_model.npz"
    ) as archive:
        assert subset == archive["core_spis"].tolist()
