import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mtsspi.companion import join_companion, load_joined_mpis
from mtsspi.corpus import ArchiveCorpus, load_inventory
from mtsspi.run_corpus import run_dataset
from mtsspi.utils import project_root

P = "phasealigned_welch_nperseg-auto"
COV = "cov_EmpiricalCovariance"


def _corpus(tmp_path: Path, archive: Path, name: str, pyspi_config: Path) -> ArchiveCorpus:
    path = tmp_path / f"{name}.yaml"
    path.write_text("\n".join([
        f"name: {name}",
        "source:",
        "  format: named-npz-v1",
        f"  archive: {archive}",
        f"  sha256: {hashlib.sha256(archive.read_bytes()).hexdigest()}",
        "  axis_order: [process, observation]",
        f"base_output_dir: {tmp_path / name}",
        f"pyspi_config: {pyspi_config}",
        "normalise: true",
        "random_seed: 1729",
        "",
    ]), encoding="utf-8")
    corpus = ArchiveCorpus.from_file(path)
    for entry in load_inventory(corpus):
        run_dataset(corpus, entry, n_jobs=1)
    return corpus


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("companion")
    rng = np.random.default_rng(0)
    members = {"first": rng.standard_normal((3, 64)), "second": rng.standard_normal((2, 80))}
    archive = tmp_path / "source.npz"
    np.savez_compressed(archive, **members,
                        __dataset_names__=np.asarray(list(members)),
                        __labels_json__=np.asarray(['["a"]', '["b"]']),
                        __shapes__=np.asarray([m.shape for m in members.values()], dtype=np.int32))
    cov_config = tmp_path / "cov.yaml"
    cov_config.write_text(".statistics.basic:\n  Covariance:\n    labels: [undirected, linear, signed]\n"
                          "    dependencies:\n    configs:\n      - estimator: EmpiricalCovariance\n",
                          encoding="utf-8")
    primary = _corpus(tmp_path, archive, "primary", cov_config)
    companion = _corpus(tmp_path, archive, "companion",
                        project_root() / "configs/pyspi/subsets/phasealigned.yaml")
    return primary.corpus_output_dir, companion.corpus_output_dir, members


def _copy(root: Path, tmp_path: Path) -> Path:
    import shutil
    return Path(shutil.copytree(root, tmp_path / root.name))


def _edit_meta(directory: Path, edit) -> None:
    path = directory / "meta.json"
    meta = json.loads(path.read_text())
    edit(meta)
    path.write_text(json.dumps(meta))


def test_join_pairs_recordings_and_appends_the_companion_spis(corpora):
    from pyspi.data import Data
    from pyspi.statistics.spectral import PhaseAlignedCorrelation

    primary, companion, members = corpora
    join = join_companion(primary, companion)
    assert len(join.pairs) == 2 and join.provenance["mixed_computation"] is False
    for (primary_dir, companion_dir), member in zip(join.pairs, members.values()):
        mpis = load_joined_mpis(primary_dir, companion_dir)
        assert list(mpis) == [COV, P]
        # P is invariant to the runner's z-scoring, so it matches the raw member.
        expected = PhaseAlignedCorrelation().multivariate(Data(member, dim_order="ps", zscore=False))
        off = ~np.eye(len(member), dtype=bool)
        np.testing.assert_allclose(mpis[P][off], expected[off], rtol=1e-12)


def test_join_refuses_different_data(corpora, tmp_path):
    primary, companion, _ = corpora
    companion = _copy(companion, tmp_path)
    first = sorted(p for p in companion.iterdir() if p.is_dir())[0]
    _edit_meta(first, lambda meta: meta["source"].update(member_sha256="0" * 64))
    with pytest.raises(ValueError, match="different data"):
        join_companion(primary, companion)


def test_join_refuses_unmatched_recordings(corpora, tmp_path):
    import shutil

    primary, companion, _ = corpora
    companion = _copy(companion, tmp_path)
    shutil.rmtree(sorted(p for p in companion.iterdir() if p.is_dir())[0])
    with pytest.raises(ValueError, match="in only one corpus"):
        join_companion(primary, companion)


def test_join_refuses_shared_spis(corpora, tmp_path):
    primary, companion, _ = corpora
    companion = _copy(companion, tmp_path)
    for directory in (p for p in companion.iterdir() if p.is_dir()):
        _edit_meta(directory, lambda meta: meta["pyspi"]["spis"].append({"name": COV}))
    with pytest.raises(ValueError, match="in both corpora"):
        join_companion(primary, companion)


def test_mixed_computation_versions_need_an_explicit_opt_in(corpora, tmp_path):
    primary, companion, _ = corpora
    companion = _copy(companion, tmp_path)
    for directory in (p for p in companion.iterdir() if p.is_dir()):
        _edit_meta(directory, lambda meta: meta["pyspi"]["version"].update(computation="0.0.0.r0"))
    with pytest.raises(ValueError, match="computation versions differ"):
        join_companion(primary, companion)
    join = join_companion(primary, companion, allow_mixed_computation=True)
    assert join.provenance["mixed_computation"] is True
    assert join.provenance["companion"]["computation"] == "0.0.0.r0"
