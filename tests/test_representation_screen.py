import numpy as np

from src.representation_screen import (
    bootstrap_group_means, classifier, evaluation_cells, fit_view, select_c, training_subsets,
)


PREPROCESSING = {"minimum_valid_fraction": .95, "variance_threshold": 1e-8,
                 "z_scaling": "center", "pca_dimensions": 3,
                 "pca_solver": "full", "pca_random_state": 1}
CLASSIFIER = {"C_grid": [.1, 1.], "inner_folds": 2, "max_iter": 500, "tolerance": 1e-4}


def test_training_budgets_are_nested_balanced_and_reproducible():
    labels = np.repeat(["a", "b", "c"], 10)
    rows = np.arange(len(labels))
    subsets = training_subsets(labels, rows, [2, 4, 8], 11)
    assert set(subsets[2]) < set(subsets[4]) < set(subsets[8])
    for n, selected in subsets.items():
        assert len(selected) == 3 * n
        np.testing.assert_array_equal(np.unique(labels[selected], return_counts=True)[1], [n] * 3)
        np.testing.assert_array_equal(selected, training_subsets(labels, rows, [n], 11)[n])


def test_test_values_cannot_change_preprocessing_or_hyperparameters():
    rng = np.random.default_rng(3)
    bank = {"m": rng.normal(size=(20, 9)), "z": rng.normal(size=(20, 12))}
    labels = np.tile(["a", "b"], 10)
    train = np.arange(12)
    before, scores = fit_view(bank, "m+z", train, PREPROCESSING)
    c, inner = select_c(bank, "m+z", train, labels, PREPROCESSING, CLASSIFIER, 11)
    bank["m"][12:] = np.nan
    bank["z"][12:] *= 1e10
    labels[12:] = "x"
    after, new_scores = fit_view(bank, "m+z", train, PREPROCESSING)
    new_c, new_inner = select_c(bank, "m+z", train, labels, PREPROCESSING, CLASSIFIER, 11)
    np.testing.assert_allclose(scores, new_scores)
    np.testing.assert_allclose(before.pca.components_, after.pca.components_)
    assert c == new_c and inner == new_inner


def test_feature_existing_only_outside_training_is_not_used():
    values = np.array([[1., np.nan], [2., np.nan], [3., np.nan], [4., 100.]])
    transformer, _ = fit_view({"m": values}, "m", np.arange(3), PREPROCESSING)
    np.testing.assert_array_equal(transformer.blocks[0].keep, [0])


def test_no_features_has_explicit_chance_baseline():
    bank = {"validity": np.ones((8, 5))}
    transform, scores = fit_view(bank, "validity", np.arange(6), PREPROCESSING)
    model = classifier(scores, np.array(["a", "b"] * 3), 1, CLASSIFIER)
    np.testing.assert_allclose(model.predict_proba(transform.transform(bank, np.array([6, 7]))), .5)


def test_train_fitted_clipping_bounds_unseen_outliers_without_test_statistics():
    from src.representation_screen import fit_block
    train = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
    config = {**PREPROCESSING, "clip_standard_deviations": 5.}
    fitted = fit_block("g", train, config)
    test = np.array([[1e10, -1e10]])
    bounded = fitted.transform(test)
    np.testing.assert_allclose(abs(bounded[0]), fitted.clip_limit / fitted.scale)
    np.testing.assert_allclose(fitted.clip_limit, 5 * train.std(axis=0))
    np.testing.assert_allclose(np.var(fitted.transform(train), axis=0).sum(), 1.)


def test_observation_partitions_and_paired_bootstrap():
    m = np.repeat([8, 16, 32], 3); t = np.tile([500, 1000, 2000], 3)
    masks = evaluation_cells(m, t, {"M": 16, "T": 1000})
    assert [int(masks[k].sum()) for k in masks] == [1, 2, 2, 4, 8]
    values = np.array([[0., 1., 0., 1.], [0., 1., 0., 1.]])
    boot = bootstrap_group_means(values, np.array(["a", "a", "b", "b"]), 100, 1)
    np.testing.assert_array_equal(boot[0] - boot[1], 0)
    # Constant paired improvement remains constant under the same bootstrap draws.
    boot = bootstrap_group_means(np.vstack((values[0], values[0] + .2)), np.array(["a", "a", "b", "b"]), 100, 1)
    np.testing.assert_allclose(boot[1] - boot[0], .2)


def test_end_to_end_screen_records_paired_groups_and_budgeted_fits(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import yaml
    from scripts.run_representation_screen import run

    config = yaml.safe_load(Path("configs/analysis/representation-stage-a-260907.yaml").read_text())
    config.update(training_instances=[0, 1, 2, 3], evaluation_instances=[10, 11],
                  labelled_realizations_per_class=[2, 4], subset_seeds=[11], representations=["m", "z"])
    config["classifier"]["C_grid"] = [.1, 1.]
    config["evaluation"]["paired_comparisons"] = [["z", "m"]]
    config["evaluation"]["bootstrap_repetitions"] = 20
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    records = []
    for label in ["a", "b"]:
        for instance in [0, 1, 2, 3, 10, 11]:
            cells = [(16, 1000)] if instance < 10 else [(m, t) for m in [8, 16, 32] for t in [500, 1000, 2000]]
            for m, t in cells:
                records.append({"label": label, "M": m, "T": t, "instance": instance,
                                "group": f"{label}|I{instance}", "row_id": f"{label}|M{m}|T{t}|I{instance}",
                                "role": "training_pool" if instance < 10 else "evaluation"})
    rng = np.random.default_rng(9)
    values = rng.normal(size=(len(records), 6))
    values[:, 0] += np.array([r["label"] == "b" for r in records]) * 5
    bank = tmp_path / "bank.npz"
    np.savez(bank, X_m=values, X_z=values.copy(), X_g=values, X_validity=np.ones_like(values),
             feature_contract="unified_ordered_v3", manifest_sha256="test-fixture",
             **{name: np.asarray([r[name] for r in records]) for name in records[0]})
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    bank.with_suffix(".json").write_text(json.dumps({"artifact_sha256": sha(bank), "manifest_sha256": "test-fixture", "protocol_sha256": sha(config_path)}))
    out = tmp_path / "result"
    run(config_path, bank, out)
    result = json.loads((out / "results.json").read_text())
    assert len(result["fits"]) == 4
    assert result["summary"]["both_M_and_T_changed"]["groups"] == 4
    assert result["summary"]["both_M_and_T_changed"]["rows"] == 16
    splits = json.loads((out / "splits.json").read_text())
    assert [len(s["row_ids"]) for s in splits["training"]] == [4, 8]
    # A separately hash-bound raw control must use precisely the same rows.
    import copy
    raw = tmp_path / "raw.npz"
    np.savez(raw, X_u=values, row_id=np.asarray([r["row_id"] for r in records]))
    raw.with_suffix(".json").write_text(json.dumps({"artifact_sha256": sha(raw), "manifest_sha256": "test-fixture"}))
    variant = copy.deepcopy(config)
    variant.update(base_protocol=str(config_path), raw_controls=str(raw), representations=["u", "z", "u+z"])
    variant["evaluation"]["paired_comparisons"] = [["u+z", "u"]]
    variant_path = tmp_path / "variant.yaml"
    variant_path.write_text(yaml.safe_dump(variant))
    run(variant_path, bank, tmp_path / "variant-result")
    additional = json.loads((tmp_path / "variant-result/results.json").read_text())
    assert additional["raw_control_input"]["sha256"] == sha(raw)
    assert len(additional["fits"]) == 6
