import json
import numpy as np
import pytest

from scripts.analyze_finite_cached_transfer import load_verified
from scripts.finite_regime_pipeline import digest


@pytest.fixture
def bank(tmp_path):
    corpus, cached, frozen = [tmp_path/name for name in ('corpus','cached','frozen')]
    for p in (corpus,cached,frozen): p.mkdir()
    np.savez(corpus/'observations.npz', row=np.ones((3,10)))
    rows=[dict(row_id='row',role='evaluation')]
    (corpus/'manifest.json').write_text(json.dumps(dict(rows=rows,archive_sha256=digest(corpus/'observations.npz'))))
    source=dict(row_id='row',execution_identity=dict(pyspi_config_sha256='p90',runner_sha256='runner',
                compute_sha256='compute',pyspi_version=dict(dist='3')))
    eligibility=dict(manifest_sha256=digest(corpus/'manifest.json'),sources=[source])
    for p in (cached,frozen): (p/'eligibility.json').write_text(json.dumps(eligibility))
    np.savez(cached/'features.npz',z=np.ones((1,3)),row_id=['row'],spi_order=['a','b','c'])
    np.savez(frozen/'model.npz',spi_order=['a','b','c'])
    return corpus,cached,frozen


def test_cache_checks_identity_without_changing_roles(bank):
    rows,z,order,sources=load_verified(*bank)
    assert rows[0]['role']=='evaluation' and z.shape==(1,3) and order==['a','b','c']


def test_changed_raw_archive_rejected(bank):
    corpus,_,_=bank
    np.savez(corpus/'observations.npz',row=np.zeros((3,10)))
    with pytest.raises(ValueError,match='observation archive'): load_verified(*bank)


def test_changed_spi_order_rejected(bank):
    _,_,frozen=bank
    np.savez(frozen/'model.npz',spi_order=['b','a','c'])
    with pytest.raises(ValueError,match='SPI order'): load_verified(*bank)


def test_changed_row_order_rejected(bank):
    _,cached,_=bank
    np.savez(cached/'features.npz',z=np.ones((1,3)),row_id=['another'],spi_order=['a','b','c'])
    with pytest.raises(ValueError,match='row order'): load_verified(*bank)
