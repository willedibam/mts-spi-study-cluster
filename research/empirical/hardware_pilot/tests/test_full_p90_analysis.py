"""Numerical and provenance checks for the full hardware analysis, without extraction."""
from dataclasses import replace
import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import yaml
import pandas as pd

from mtsspi.corpus import ArchiveCorpus
from mtsspi.provenance import array_sha256, file_sha256
from mtsspi.spi_spi_contract import build_unified_features
from research.empirical.hardware_pilot.calibrate_hardware_pilot import lag_matrix
from research.empirical.hardware_pilot.full_p90_analysis import (
    FullP90, LAG_PAIR, contrast_contributions, edge_diagnostics, fit_pca,
    frozen_benchmark, load_run, loading_stability, partition_covariance,
)


def test_covariance_partition_exposes_positive_mixture_of_negative_groups(tmp_path):
    edge = ~np.eye(3, dtype=bool)
    a, b = np.full((3,3),np.nan), np.full((3,3),np.nan)
    a[edge], b[edge] = [0,1,2,10,11,12], [2,1,0,12,11,10]
    path = tmp_path/'mixture.npz'; np.savez(path,a=a,b=b)
    feature = build_unified_features(dict(a=a,b=b),['a','b'])
    bank = FullP90(tmp_path,({'name':'mixture','topology':'ER','step':0},),feature.z[None],
                   feature.schema,(path,),(),{})
    group = np.array([True]*3+[False]*3)
    row = partition_covariance(bank,0,[0],{0:group}).iloc[0]
    assert row.z > .9 and row.within_coherent < 0 and row.within_other < 0
    assert row['between'] > row.z
    assert_allclose(row.within_coherent+row.within_other+row['between'],np.corrcoef(a[edge],b[edge])[0,1],atol=1e-14)
    with pytest.raises(ValueError,match='nonempty'):
        partition_covariance(bank,0,[0],{0:np.ones(6,bool)})


def test_er_fit_ignores_sw_values_and_missingness_without_imputation():
    z = np.array([[-.8, .2, -.3, -.6], [-.2, .2, np.nan, -.2],
                  [.2, .2, .1, .2], [.8, .2, .2, .6],
                  [.99, .9, .9, np.nan], [-.99, -.9, -.9, .1]])
    er = np.array([True] * 4 + [False] * 2)
    model = fit_pca(z, er)
    assert_array_equal(model.features, [0, 3])
    assert_allclose(model.center, z[:4, [0, 3]].mean(axis=0), atol=0, rtol=0)
    changed = z.copy()
    changed[~er] = [[-.4, np.nan, np.nan, .8], [.7, .1, .3, -.5]]
    refit = fit_pca(changed, er)
    assert_array_equal(refit.features, model.features)
    assert_allclose(refit.center, model.center, atol=0, rtol=0)
    assert_allclose(refit.components, model.components, atol=0, rtol=0)
    scores = model.project(z)
    assert np.isnan(scores[4]).all()
    assert np.isfinite(scores[[0, 1, 2, 3, 5]]).all()
    assert_allclose(model.missing_fraction(z), [0, 0, 0, 0, .5, 0])
    assert_allclose(scores[5], (z[5, model.features] - model.center) @ model.components.T)
    common = np.isfinite(z).all(axis=0)
    # Requiring common completeness leaves only one varying feature; no silent fallback.
    with pytest.raises(ValueError, match="Insufficient varying"):
        fit_pca(z, er, feature_mask=common)


def test_center_only_pca_preserves_feature_scale_and_loading_sign():
    z = np.array([[-.9, -.03], [-.3, .03], [.3, .03], [.9, -.03]])
    model = fit_pca(z, np.ones(4, bool))
    assert_allclose(model.components[0], [1., 0.], atol=1e-15)
    assert model.variance_ratio[0] > .99
    assert_allclose(model.project(z)[:, 0], z[:, 0], atol=1e-15)
    with pytest.raises(ValueError, match="bounded"):
        fit_pca(z * 3, np.ones(4, bool))
    proportional = np.column_stack((z[:, 0], z[:, 0] / 10))
    aligned = fit_pca(proportional, np.ones(4, bool))
    assert_allclose(loading_stability(proportional, aligned), 1., atol=1e-15)


def test_original_directed_edges_match_contract_and_cached_deletion(tmp_path):
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=(2, 5, 5))
    np.fill_diagonal(a, np.nan)
    np.fill_diagonal(b, np.nan)
    path = tmp_path / "mpis.npz"
    np.savez(path, a=a, b=b)
    feature = build_unified_features(dict(a=a, b=b), ["a", "b"])
    bank = FullP90(tmp_path, ({"name": "synthetic"},), feature.z[None], feature.schema, (path,), (), {})
    result = edge_diagnostics(bank, 0).iloc[0]
    edge = ~np.eye(5, dtype=bool)
    assert_allclose(result.z, np.corrcoef(a[edge], b[edge])[0, 1], atol=1e-14)
    assert_allclose(result.z, feature.z[0], atol=5e-8)
    legacy = np.corrcoef((a + a.T)[edge], (b + b.T)[edge])[0, 1]
    assert abs(legacy - result.z) > .01
    deleted = []
    for channel in range(5):
        mask = edge.copy()
        mask[channel, :] = mask[:, channel] = False
        deleted.append(np.corrcoef(a[mask], b[mask])[0, 1])
    assert_allclose([result.delete_min, result.delete_max], [min(deleted), max(deleted)], atol=1e-14)
    assert result.undefined_deletions == 0
    np.savez(path, a=np.ones((5, 5)), b=b)
    collapsed = edge_diagnostics(bank, 0).iloc[0]
    assert np.isnan(collapsed.z) and collapsed.undefined_deletions == 5


def test_contributions_sum_to_pc1_change(tmp_path):
    rng = np.random.default_rng(88)
    z = rng.uniform(-.9, .9, (6, 3))
    schema = build_unified_features({s: rng.normal(size=(4, 4)) for s in "abc"}, list("abc")).schema
    bank = FullP90(tmp_path, (), z, schema, (), (), {})
    model = fit_pca(z, np.array([True] * 4 + [False] * 2))
    table = contrast_contributions(bank, model, 0, 5)
    scores = model.project(z)
    assert_allclose(table.pc1_contribution.sum(), scores[5, 0] - scores[0, 0], atol=1e-14)
    missing = z.copy()
    missing[5, 0] = np.nan
    with pytest.raises(ValueError, match="unprojectable"):
        contrast_contributions(replace(bank, z=missing), model, 0, 5)


def _saved_run(root):
    """Small synthetic catalogue with one failed SPI and a remote execution identity."""
    rng = np.random.default_rng(20)
    x = rng.normal(size=(28, 5000))
    np.savez(root / "observations.npz", record=x, __dataset_names__=np.array(["record"]),
             __labels_json__=np.array(['["ER"]']), __shapes__=np.array([[28, 5000]]),
             __axis_order__=np.array(["process", "observation"]))
    archive_hash = file_sha256(root / "observations.npz")
    catalogue_path = root / "p90.yaml"
    catalogue_path.write_text("synthetic: true\n")
    config = dict(name="synthetic", source=dict(format="named-npz-v1", archive="/remote/observations.npz",
                                               sha256=archive_hash, axis_order=["process", "observation"]),
                  base_output_dir="/remote/mpis", pyspi_config=str(catalogue_path), normalise=False, random_seed=77)
    config_path = root / "external.yaml"
    config_path.write_text(yaml.safe_dump(config))
    identity = dict(corpus_config_sha256=file_sha256(config_path), pyspi_config_sha256=file_sha256(catalogue_path),
                    code_sha256={name: "a" * 64 for name in ("run_corpus.py", "corpus.py", "generated_corpus.py", "compute.py")},
                    pyspi_version=dict(dist="3.0.0", computation="3.0.0.r9", source="/remote/frozen-pyspi"))
    (root / "execution-identity.json").write_text(json.dumps(identity))
    row = dict(name="record", topology="ER", M=28, T=5000, fs=500, seconds=10, member_sha256=array_sha256(x))
    (root / "manifest.json").write_text(json.dumps(dict(archive_sha256=archive_hash, rows=[row])))
    corpus = replace(ArchiveCorpus.from_file(config_path), archive=root / "observations.npz", base_output_dir=root / "mpis")
    entry, = corpus.entries()
    path = entry.output_dir(corpus)
    path.mkdir(parents=True)
    a, b = rng.normal(size=(2, 28, 28))
    np.savez(path / "spi_mpis.npz", a=a, b=b, failed=np.full((28, 28), np.nan))
    metadata = dict(status="complete", **corpus.describe(entry, {"member_sha256": array_sha256(x)}),
                    job=dict(index=1, n_jobs=4), execution_identity=identity, normalise=False, random_seed=77,
                    experiment=dict(git_commit="unknown", git_dirty=None),
                    pyspi=dict(version=identity["pyspi_version"], errors={"failed": "Undefined statistic"},
                               spis=[dict(name=name) for name in ("a", "b", "failed")]))
    (path / "meta.json").write_text(json.dumps(metadata))
    return path, metadata


def test_saved_identity_relocates_paths_keeps_failures_and_rejects_mixed_versions(tmp_path):
    path, metadata = _saved_run(tmp_path)
    bank = load_run(tmp_path, expected_spis=3, min_coverage=.60)
    assert bank.z.shape == (1, 3)
    assert np.isfinite(bank.z[0, 0]) and np.isnan(bank.z[0, 1:]).all()
    assert bank.sources[0]["failed_spis"] == {"failed": "Undefined statistic"}
    metadata["execution_identity"]["pyspi_version"]["computation"] = "3.0.0.r7"
    (path / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="execution identity mismatch"):
        load_run(tmp_path, expected_spis=3, min_coverage=.60)


def test_production_gate_rejects_inadequate_finite_edge_coverage(tmp_path):
    _saved_run(tmp_path)
    with pytest.raises(ValueError, match="Insufficient SPI coverage.*0.666667"):
        load_run(tmp_path, expected_spis=3)


def test_production_gate_rejects_more_than_fifteen_failed_spis(tmp_path):
    path, metadata = _saved_run(tmp_path)
    with np.load(path / "spi_mpis.npz") as archive:
        matrices = {name: archive[name] for name in archive.files}
    for index in range(15):
        name = f"additional-failure-{index}"
        matrices[name] = np.full((28, 28), np.nan)
        metadata["pyspi"]["spis"].append(dict(name=name))
        metadata["pyspi"]["errors"][name] = "Undefined statistic"
    np.savez(path / "spi_mpis.npz", **matrices)
    (path / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="16 failed SPIs"):
        load_run(tmp_path, expected_spis=18, min_coverage=0.)


@pytest.mark.parametrize("workers", [1, None])
def test_production_gate_requires_recorded_four_workers(tmp_path, workers):
    path, metadata = _saved_run(tmp_path)
    if workers is None:
        del metadata["job"]["n_jobs"]
    else:
        metadata["job"]["n_jobs"] = workers
    (path / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="expected n_jobs=4"):
        load_run(tmp_path, expected_spis=3, min_coverage=.60)


def test_saved_config_and_source_member_hashes_are_mandatory(tmp_path):
    _saved_run(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0]["member_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source member hash mismatch"):
        load_run(tmp_path, expected_spis=3)
    with (tmp_path / "external.yaml").open("a") as handle:
        handle.write("\n# changed after execution\n")
    with pytest.raises(ValueError, match="Frozen content hash mismatch"):
        load_run(tmp_path, expected_spis=3)


def test_frozen_lag_recomputation_uses_full_record_archive(tmp_path):
    rng = np.random.default_rng(442)
    t = np.arange(5000) / 500
    phase = rng.uniform(-.4, .4, (28, 1))
    x = np.cos(2 * np.pi * 116 * t + phase) + .25 * np.cos(2 * np.pi * 58 * t + 3 * phase)
    x += .02 * rng.normal(size=x.shape)
    np.savez(tmp_path / "observations.npz", example=x)
    matrices = {name: lag_matrix(x, tau) for name, tau in zip(LAG_PAIR, (1, 10), strict=True)}
    path = tmp_path / "mpi.npz"
    np.savez(path, **matrices)
    features = build_unified_features(matrices, LAG_PAIR)
    bank = FullP90(tmp_path, (dict(name="example", topology="SW", step=0, fs=500),),
                   features.z[None], features.schema, (path,), (), {})
    result = frozen_benchmark(bank).iloc[0]
    assert result.archive_parity_pass
    assert result.archive_profile_max_error == 0
    assert_allclose(result.archived_z, features.z[0], atol=5e-8)
    assert_allclose(result.archived_z, result.z_full_record, atol=1e-14)
    # A bad archive is reported as a failed numerical check, not silently accepted.
    matrices[LAG_PAIR[0]][0, 1] += .1
    np.savez(path, **matrices)
    changed = frozen_benchmark(bank).iloc[0]
    assert not changed.archive_parity_pass
    assert_allclose(changed.archive_profile_max_error, .1, atol=1e-14)


def test_unavailable_sensitivity_does_not_discard_primary_fit(tmp_path, monkeypatch):
    from research.empirical.hardware_pilot import full_p90_analysis as analysis
    rng = np.random.default_rng(12)
    names = ["corr_pearson_tau-1", "corr_pearson_tau-2", "corr_pearson_tau-10"]
    schema = build_unified_features({s: rng.normal(size=(3, 3)) for s in names}, names).schema
    z = np.array([[-.9, -.2, .1], [-.3, .2, .1], [.3, -.2, .1], [.9, .2, .1],
                  [.8, np.nan, .5], [.7, -.1, .5]])
    rows = tuple(dict(name=f"record-{i}", topology="ER" if i < 4 else "SW", step=i, voltage=i * .005)
                 for i in range(len(z)))
    bank = FullP90(tmp_path, rows, z, schema, (), (), {"pyspi_version": {"computation": "3.0.0.r9"}})
    monkeypatch.setattr(analysis, "load_run", lambda _: bank)
    monkeypatch.setattr(analysis, "frozen_benchmark", lambda _: pd.DataFrame({
        "name": [r["name"] for r in rows], "archive_profile_max_error": np.zeros(len(z)),
        "archive_parity_pass": np.ones(len(z), bool), "reconstruction_target_met": np.ones(len(z), bool)}))
    result = analysis.analyse(tmp_path)
    assert set(result["models"]) == {"er"}
    assert set(result["summary"]["unavailable_sensitivities"]) == {"er_common_complete", "er_no_double_correlation"}
    assert result["summary"]["models"]["er"]["unprojectable"] == ["record-4"]
    scores = pd.read_csv(tmp_path / "analysis" / "scores.csv")
    assert len(scores) == 6 and np.isnan(scores.loc[4, "er_pc1"])
    assert scores.loc[4, "er_missing_fraction"] == .5
    summary_text = (tmp_path / "analysis" / "summary.json").read_text()
    assert "NaN" not in summary_text
