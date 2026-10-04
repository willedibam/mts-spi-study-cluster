"""Audit arbitrary completed index groups without assuming a corpus prefix."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_pearson_size_match import DATA, OUT, ROOT, RUN
from scripts.spi_baseline_exploration import sha
from src.run_external_corpus import _array_sha256
from src.spi_spi_contract import build_unified_feature_values
from src.utils import slugify


def audit(indices, name):
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert sha(DATA/'observations.npz') == manifest['archive_sha256']
    result, order = [], None
    with np.load(DATA/'observations.npz') as raw:
        for index in indices:
            r = manifest['rows'][index-1]
            folder = DATA/'mpis'/RUN/f"{index:04d}-{slugify(r['row_id'],'dataset')}"
            meta = json.loads((folder/'meta.json').read_text())
            assert meta['status'] == 'complete' and meta['dataset_name'] == r['row_id']
            assert (meta['M'], meta['T']) == (r['M'], r['T'])
            assert meta['source']['archive_sha256'] == manifest['archive_sha256']
            assert meta['source']['member_sha256'] == _array_sha256(raw[r['row_id']])
            assert meta['experiment']['git_commit'].startswith('a770dba') and not meta['experiment']['git_dirty']
            assert meta['random_seed'] == 261011 and meta['job']['estimator_rng_policy'] == 'isolated_serial'
            assert meta['pyspi']['version']['computation'] == '3.0.0.r7'
            for key, path in [('corpus_config_sha256', f'configs/external/{RUN}.yaml'),
                              ('pyspi_config_sha256', 'configs/pyspi/benchmarked_p90.yaml'),
                              ('runner_sha256', 'src/run_external_corpus.py'), ('compute_sha256', 'src/compute.py')]:
                assert meta['execution_identity'][key] == sha(ROOT/path)
            names = [s['name'] for s in meta['pyspi']['spis']]
            if order is None:
                order = names
            assert names == order and len(names) == 289
            with np.load(folder/'spi_mpis.npz') as a:
                assert set(a.files) == set(names)
                matrices = {k: a[k] for k in names}
            assert all(a.shape == (r['M'], r['M']) for a in matrices.values())
            mask = ~np.eye(r['M'], dtype=bool)
            c = matrices['cov_EmpiricalCovariance']
            np.testing.assert_allclose(c[mask], np.cov(raw[r['row_id']], bias=True)[mask], atol=1e-12)
            np.testing.assert_allclose([c[mask].mean(), abs(c[mask]).mean()],
                                      [r['mean_covariance'], r['mean_abs_Pearson']], atol=1e-12)
            incomplete = [k for k in names if not np.isfinite(matrices[k][mask]).all()]
            z, _, _ = build_unified_feature_values(matrices, names)
            result.append(dict(index=index, row_id=r['row_id'], M=r['M'], T=r['T'],
                seconds=meta['job']['compute_seconds'], finite_mean_profiles=289-len(incomplete),
                finite_z=int(np.isfinite(z).sum()), incomplete=incomplete,
                mpi_sha256=sha(folder/'spi_mpis.npz'), meta_sha256=sha(folder/'meta.json')))
    target = OUT/'audits'
    target.mkdir(exist_ok=True)
    summary = dict(complete=len(result), indices=indices, records=result,
        compute_seconds_quantiles=dict(zip(['min','median','p95','max'],
            np.quantile([r['seconds'] for r in result], [0,.5,.95,1]).tolist())),
        finite_mean_profile_range=[min(r['finite_mean_profiles'] for r in result), max(r['finite_mean_profiles'] for r in result)])
    (target/f'{name}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['records','indices']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    with open(args.indices) as f:
        selected = [int(line) for line in f if line.strip()]
    assert len(selected) == len(set(selected)) and selected
    with threadpool_limits(limits=4):
        audit(selected, args.name)
