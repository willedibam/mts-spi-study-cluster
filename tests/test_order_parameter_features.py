import json
from pathlib import Path

import numpy as np

from src.order_parameter_features import (
    build_meta_feature_matrix,
    explicit_phase_spi_names,
    stable_spi_names,
    validate_spi_catalogs,
)


CATALOG = [
    {"name": "a", "directed": False, "class_name": "Correlation"},
    {"name": "b", "directed": False, "class_name": "MutualInformation"},
    {"name": "plv", "directed": False, "class_name": "PhaseLockingValue"},
]


def _write_dataset(path: Path, *, invalid_b: bool = False) -> None:
    path.mkdir()
    (path / "meta.json").write_text(
        json.dumps({"pyspi": {"spis": CATALOG}}), encoding="utf-8"
    )
    a = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    b = np.ones((3, 3)) if invalid_b else a**2
    np.savez(path / "spi_mpis.npz", a=a, b=b, plv=np.sqrt(a))


def test_stability_selection_and_phase_ablation(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_dataset(first)
    _write_dataset(second, invalid_b=True)
    catalog = validate_spi_catalogs([first, second])
    names, rates = stable_spi_names([first, second], catalog)
    assert names == ["a", "plv"]
    assert rates["b"] == 0.5
    assert explicit_phase_spi_names(catalog) == {"plv"}


def test_meta_features_preserve_invalid_values_for_explicit_imputation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    matrix, pairs = build_meta_feature_matrix(
        [dataset], CATALOG, ["a", "b", "plv"], metric="pearson"
    )
    assert matrix.shape == (1, 3)
    assert pairs == [("a", "b"), ("a", "plv"), ("b", "plv")]
    assert np.isfinite(matrix).all()
