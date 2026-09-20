import json

import numpy as np
import pytest

from scripts import tasep_confirmation as confirmation


@pytest.fixture
def sealed(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    np.savez_compressed(frozen / "model.npz", keep=np.array([0]), impute=np.array([0.]), center=np.array([0.]),
        component=np.array([1.]), score_scale=np.asarray(1.), spi_order=np.array(["a", "b"]))
    (frozen / "summary.json").write_text(json.dumps(dict(passes=True, display_sign=-1)))
    (frozen / "geometry.json").write_text(json.dumps(dict(passes=True)))
    gate = tmp_path / "pilot-physics.json"
    gate.write_text(json.dumps(dict(system="tasep", arms={"32": dict(passes=True)},
                                    source_sha256=confirmation.digest(confirmation.generator.__file__))))
    path = tmp_path / "seal.json"
    plan = confirmation.create_seal(frozen, path, gate)
    return frozen, path, plan


def test_sealed_fresh_full_state_design(sealed):
    frozen, path, plan = sealed
    assert len(plan["cases"]) == 672
    assert {c["N"] for c in plan["cases"]} == {32}
    assert {c["seed"] for c in plan["cases"]} == set(range(2609152001, 2609152033))
    assert {c["role"] for c in plan["cases"]} == {"evaluation"}
    assert sorted({c["alpha"] for c in plan["cases"]}) == np.round(np.linspace(.15, .25, 21), 8).tolist()
    assert confirmation.read_seal(path, frozen) == plan


def test_model_and_gate_tampering_rejected(sealed):
    frozen, path, plan = sealed
    (frozen / "summary.json").write_text(json.dumps(dict(passes=True, display_sign=1)))
    with pytest.raises(ValueError, match="frozen model"):
        confirmation.read_seal(path, frozen)
    plan["physical_gates"]["maximum_exact_error_p95"] = .5
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="seal contents"):
        confirmation.read_seal(path)


def test_physics_retains_one_archive_and_event_weighted_blocks(sealed, tmp_path, monkeypatch):
    _, seal, plan = sealed
    calls = []
    def fake_simulate(case, protocol):
        calls.append((case, protocol))
        return dict(observed=np.tile(np.arange(2000) % 2, (32, 1)).astype(np.uint8), reference_density=np.ones(100),
                    final_state=np.zeros(32, dtype=np.uint8)), dict(Q_blocks=[.3] * 32, Q_reference=.3, elapsed_seconds=.01)
    monkeypatch.setattr(confirmation.generator, "simulate", fake_simulate)
    root = tmp_path / "physics"
    path = confirmation.run_physics(0, root, seal)
    assert len(list(root.iterdir())) == 1
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["role"] == "evaluation" and metadata["seal_identity"] == plan["seal_identity"]
        assert archive["reference_block_density"].mean() == pytest.approx(.3)
        assert archive["reference_density"].mean() == 1
    assert calls == [(dict(N=32, alpha=.15, seed=2609152001), confirmation.PROTOCOL)]
    with pytest.raises(FileExistsError):
        confirmation.run_physics(0, root, seal)


def test_physical_reductions_use_blocks_and_exact_density():
    blocks = np.linspace(.45, .55, 32)
    exact = dict(density=.5, current=.16)
    metadata = dict(Q_blocks=blocks.tolist(), Q_reference=blocks.mean(), Q_exact=.5, exact_current=.16,
        Q_first_half=blocks[:16].mean(), Q_second_half=blocks[16:].mean(),
        block_mean_se=blocks.std(ddof=1)/np.sqrt(32), reference_absolute_error=abs(blocks.mean()-.5))
    assert confirmation.validate_physical_summaries(metadata, blocks, exact)["Q_reference"] == pytest.approx(.5)
    metadata["Q_reference"] = .7
    with pytest.raises(ValueError, match="Q_reference"):
        confirmation.validate_physical_summaries(metadata, blocks, exact)


def test_pilot_physical_thresholds_preserved_without_exclusions():
    records = [dict(control=alpha, Q=q, raw_ok=True, reference_error=.001, half_difference=.001)
               for alpha, q in ((.15, .2), (.2, .5), (.25, .8))]
    result = confirmation.evaluate_physics(records)
    assert result["passes"] and result["excluded_rows"] == 0
    records[0]["raw_ok"] = False
    assert not confirmation.evaluate_physics(records)["passes"]
    records[0]["raw_ok"] = True
    records[0]["half_difference"] = .15
    assert not confirmation.evaluate_physics(records)["checks"]["reference_stability"]
    records[0]["half_difference"] = .001
    records[0]["reference_error"] = .1
    assert not confirmation.evaluate_physics(records)["checks"]["exact_agreement"]


def test_numeric_model_array_change_rejected(sealed):
    frozen, path, _ = sealed
    with np.load(frozen / "model.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["center"] = np.array([.01])
    np.savez_compressed(frozen / "model.npz", **arrays)
    with pytest.raises(ValueError, match="frozen model"):
        confirmation.read_seal(path, frozen)
