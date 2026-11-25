"""Feature bank of an archive corpus: per-SPI means, distribution summaries and the unified ordered z row.

Restored from `research/proof/baselines/analyze_native_coupling.extract` (removed in ea7f9d43), with the
catalogue taken from the recordings instead of the frozen 289-SPI YAML: every recording must share one
SPI order, catalogue hash and pyspi computation version, which `sources.json` records.
"""
import json
from pathlib import Path

import numpy as np

from mtsspi.provenance import array_sha256
from mtsspi.spi_spi_contract import build_unified_feature_values
from mtsspi.utils import slugify
from research.proof.p90.spi_baseline_exploration import sha, summarize


def extract(data, out, corpus, mpis=None, row_limit=None):
    """Write `features.npz` and `sources.json` to `out` from `<mpis>/<corpus>/NNNN-<row_id>/` (default `data/mpis`)."""
    data, out = Path(data), Path(out)
    mpis = Path(mpis) if mpis else data/'mpis'
    manifest = json.loads((data/'manifest.json').read_text()); rows = manifest['rows'][:row_limit]
    means, distributions, zs, validity, profile_validity, sources = [], [], [], [], [], []
    order = identity = None
    with np.load(data/'observations.npz') as raw:
        assert sha(data/'observations.npz') == manifest['archive_sha256']
        for r in rows:
            folder = mpis/corpus/f"{r['corpus_index']+1:04d}-{slugify(r['row_id'], 'dataset')}"
            meta = json.loads((folder/'meta.json').read_text()); p = folder/'spi_mpis.npz'
            assert meta['status'] == 'complete' and meta['dataset_name'] == r['row_id']
            assert (meta['M'], meta['T']) == (r['M'], r['T'])
            assert meta['source']['archive_sha256'] == manifest['archive_sha256']
            assert meta['source']['member_sha256'] == array_sha256(raw[r['row_id']])
            names = [x['name'] for x in meta['pyspi']['spis']]
            run = (meta['pyspi']['config_sha256'], meta['pyspi']['version']['computation'], meta['pyspi']['version']['dist'],
                   meta['job'].get('estimator_rng_policy'))
            if order is None: order, identity = names, run
            assert order == names and identity == run, f"mixed catalogue or pyspi in {folder.name}"
            with np.load(p) as a:
                assert set(a.files) == set(order)
                values = {k: a[k] for k in order}
            assert all(v.shape == (r['M'], r['M']) for v in values.values())
            marginal = summarize(values, order)
            z, _, invalid = build_unified_feature_values(values, order)
            means.append(marginal[:, 0]); distributions.append(marginal.reshape(-1)); zs.append(z)
            validity.append(np.isfinite(marginal[:, 0]).astype(float))
            profile_validity.append(np.array([k not in invalid for k in order], float))
            sources.append(dict(row_id=r['row_id'], mpi_sha256=sha(p), meta_sha256=sha(folder/'meta.json'),
                                compute_seconds=meta['job']['compute_seconds']))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'features.npz', mean=means, distribution=distributions, z=zs, validity=validity,
                        profile_validity=profile_validity, z_validity=np.isfinite(zs).astype(np.uint8),
                        row_id=np.array([r['row_id'] for r in rows]), spi_order=np.array(order))
    catalogue_sha256, computation, dist, rng_policy = identity
    (out/'sources.json').write_text(json.dumps(dict(
        sources=sources, manifest_sha256=sha(data/'manifest.json'), features_sha256=sha(out/'features.npz'),
        catalogue=dict(n_spis=len(order), config_sha256=catalogue_sha256, computation=computation, dist=dist,
                       estimator_rng_policy=rng_policy),
        code_sha256=sha(Path(__file__))), indent=2)+'\n')
