from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import yaml

from scripts.analyze_cross_mt_transfer import analyze
from scripts.freeze_cross_mt_protocol import freeze


class _FakeUMAP:
    def __init__(self, **_: object) -> None:
        self.embedding_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "_FakeUMAP":
        self.embedding_ = values[:, :2]
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values[:, :2]


def _write_artifact(
    path: Path,
    data_root: Path,
    classes: list[str],
    instances: list[int],
) -> None:
    rows: list[tuple[str, int, int, int, str, np.random.Generator]] = []
    for label in classes:
        for M in (2, 3):
            for T in (20, 30):
                for instance in instances:
                    dataset = data_root / f"{label}-M{M}-T{T}-I{instance}"
                    dataset.mkdir(parents=True)
                    rng = np.random.default_rng(
                        1000 * (label == "b") + 100 * M + T + instance
                    )
                    np.save(
                        dataset / "timeseries.npy",
                        rng.normal(loc=3 * (label == "b"), size=(T, M)),
                    )
                    rows.append((label, M, T, instance, str(dataset), rng))
    X_sym = np.vstack(
        [row[-1].normal(loc=2 * (row[0] == "b"), size=3) for row in rows]
    ).astype(np.float32)
    X_dir = np.vstack(
        [row[-1].normal(loc=row[0] == "b", size=3) for row in rows]
    ).astype(np.float32)
    np.savez_compressed(
        path,
        feature_contract="direction_preserving_v2",
        metric="spearman",
        schema_sha256="schema",
        sym_schema_sha256="sym",
        dir_schema_sha256="dir",
        pyspi_provenance_json='{"status":"complete","version":"3.0.0.r7","config_sha256":"cfg"}',
        y=np.asarray([row[0] for row in rows]),
        dataset_paths=np.asarray([row[4] for row in rows], dtype=object),
        M=np.asarray([row[1] for row in rows]),
        T=np.asarray([row[2] for row in rows]),
        instance=np.asarray([row[3] for row in rows]),
        X_sym=X_sym,
        X_dir=X_dir,
        spi_order=np.asarray(["s0", "s1", "s2"], dtype=object),
        directed_flags=np.asarray([False, True, False]),
        feature_block=np.asarray(["sym"] * 3 + ["dir"] * 3, dtype=object),
        feature_relation=np.asarray(
            ["sym"] * 3 + ["parallel", "parallel", "reciprocity"], dtype=object
        ),
        feature_spi_a=np.asarray(["s0", "s0", "s1", "s0", "s1", "s1"], dtype=object),
        feature_spi_b=np.asarray(["s1", "s2", "s2", "s1", "s2", "s1"], dtype=object),
    )


def test_frozen_pipeline_evaluates_confirmation_without_refitting(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = {
        "study_id": "test",
        "feature_contract": "direction_preserving_v2",
        "metric": "spearman",
        "expected_pyspi_computation": "3.0.0.r7",
        "expected_pyspi_config_sha256": "cfg",
        "classes": ["a", "b"],
        "M_values": [2, 3],
        "T_values": [20, 30],
        "development_instances": [0, 1],
        "confirmation_instances": [2, 3],
        "representations": {"primary": "sym", "sensitivities": ["dir", "augmented_balanced"]},
        "preprocessing": {"minimum_valid_fraction": 0.95, "variance_threshold": 1e-8},
        "projection": {"dimensions": 3, "random_state": 7},
        "classification": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tolerance": 1e-4,
        },
        "uncertainty": {
            "bootstrap_repetitions": 10,
            "confidence_level": 0.9,
            "permutation_repetitions": 10,
            "random_state": 7,
        },
        "illustration": {
            "umap": {"n_neighbors": 3, "min_dist": 0.1, "metric": "euclidean", "random_state": 7}
        },
        "development_evidence": {
            "cml_panel": {
                "classes": ["a", "b"],
                "training_instances": [0],
                "evaluation_instances": [1],
            }
        },
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    proof = tmp_path / "proof.npz"
    cml = tmp_path / "cml.npz"
    confirmation = tmp_path / "confirmation.npz"
    _write_artifact(proof, tmp_path / "proof-data", ["a"], [0, 1])
    _write_artifact(cml, tmp_path / "cml-data", ["b"], [0, 1])
    _write_artifact(confirmation, tmp_path / "confirmation-data", ["a", "b"], [2, 3])

    manifest = freeze(
        SimpleNamespace(
            protocol=str(protocol_path),
            proof_features=str(proof),
            cml_development_features=str(cml),
            baseline_cache=str(tmp_path / "development-baselines.npz"),
            model_bundle=str(tmp_path / "models.joblib"),
            output=str(tmp_path / "manifest.json"),
            workers=1,
        )
    )
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=_FakeUMAP))
    result = analyze(
        SimpleNamespace(
            protocol=str(protocol_path),
            manifest=str(tmp_path / "manifest.json"),
            model_bundle=str(tmp_path / "models.joblib"),
            confirmation_features=str(confirmation),
            baseline_cache=str(tmp_path / "confirmation-baselines.npz"),
            coordinates_output=str(tmp_path / "coordinates.npz"),
            output=str(tmp_path / "results.json"),
            workers=1,
        )
    )

    assert manifest["status"] == "development_frozen_confirmation_unseen"
    assert manifest["rows"] == result["rows"] == 16
    assert result["status"] == "confirmation_evaluated"
    assert result["umap_status"] == "complete_development_fit_confirmation_transform"
    assert set(result["results"]) == {
        "sym",
        "dir",
        "augmented_balanced",
        "pooled_univariate",
        "pooled_dependence",
        "pooled_combined",
    }
    assert "both_M_and_T_changed" in result["results"]["sym"]["retrieval"]
