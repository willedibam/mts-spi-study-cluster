import json

import numpy as np
import pytest

from scripts import rossler_confirmation as confirmation


@pytest.fixture
def sealed(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    np.savez_compressed(frozen / "model.npz", keep=np.array([0]), impute=np.array([0.]),
        center=np.array([0.]), component=np.array([1.]), score_scale=np.asarray(1.), spi_order=np.array(["a", "b"]))
    (frozen / "geometry.json").write_text(json.dumps(dict(passes=True)))
    (frozen / "summary.json").write_text(json.dumps(dict(passes=True, display_sign=1)))
    gate = tmp_path / "pilot-gate.json"
    gate.write_text(json.dumps(dict(passes=True, checks=dict(timestep_check=True),
        source_sha256=confirmation.digest(confirmation.generator.__file__), maximum_anchor_dt_difference=.0001)))
    path = tmp_path / "seal.json"
    plan = confirmation.create_seal(frozen, path, gate)
    return frozen, path, plan


def test_fresh_grid_and_model_arrays_sealed(sealed):
    frozen, path, plan = sealed
    assert len(plan["cases"]) == 672
    assert {row["seed"] for row in plan["cases"]} == set(range(2609151001, 2609151033))
    assert {row["role"] for row in plan["cases"]} == {"evaluation"}
    assert len({row["coupling"] for row in plan["cases"]}) == 21
    assert set(plan["frozen_identity"]["arrays"]) == set(confirmation.MODEL_ARRAYS)
    assert confirmation.read_seal(path, frozen) == plan


def test_seal_and_frozen_sign_tampering_rejected(sealed):
    frozen, path, plan = sealed
    (frozen / "summary.json").write_text(json.dumps(dict(passes=True, display_sign=-1)))
    with pytest.raises(ValueError, match="frozen model"):
        confirmation.read_seal(path, frozen)
    plan["cases"][0]["seed"] = 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="seal contents"):
        confirmation.read_seal(path)


def test_physics_single_archive_embedded_metadata_and_no_overwrite(sealed, tmp_path, monkeypatch):
    _, seal, plan = sealed
    calls = []
    def fake_simulate(coupling, seed, **kwargs):
        calls.append((coupling, seed, kwargs))
        return dict(X=np.ones((6, 2000))), dict(reference_summary=dict(Q=.02), elapsed_seconds=.1)
    monkeypatch.setattr(confirmation.generator, "simulate", fake_simulate)
    root = tmp_path / "physics"
    path = confirmation.run_physics(0, root, seal)
    assert len(list(root.iterdir())) == 1
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"]))
        assert meta["role"] == "evaluation" and meta["seal_identity"] == plan["seal_identity"]
    assert calls[0] == (.015, 2609151001, confirmation.PROTOCOL)
    with pytest.raises(FileExistsError):
        confirmation.run_physics(0, root, seal)


def test_prospective_physics_gates_stop_raw_failure_without_exclusions(sealed):
    _, _, plan = sealed
    rows = [dict(control=c, Q=q, half_difference=.0001, raw_ok=True, finite_summary=True,
                 minimum_radius=2., maximum_phase_increment=.02, poincare_discrepancy=.00001)
            for c, q in ((.015, .027), (.04, .00001))]
    result = confirmation.evaluate_physics(rows, plan["physical_gates"])
    assert result["passes"] and result["excluded_rows"] == 0
    rows[0]["raw_ok"] = False
    assert not confirmation.evaluate_physics(rows, plan["physical_gates"])["passes"]
    rows[0]["raw_ok"] = True
    rows[0]["half_difference"] = .1
    assert not confirmation.evaluate_physics(rows, plan["physical_gates"])["checks"]["reference_stability"]


def test_model_numeric_change_is_bound(sealed):
    frozen, path, _ = sealed
    with np.load(frozen / "model.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["component"] = np.array([-1.])
    np.savez_compressed(frozen / "model.npz", **arrays)
    with pytest.raises(ValueError, match="frozen model"):
        confirmation.read_seal(path, frozen)
