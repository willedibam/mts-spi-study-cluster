"""Audit downloaded p90 outputs against immutable data, code and RNG bindings."""
import argparse,json
import numpy as np
from scripts.build_lag_surrogate import DATA,OUT,ROOT,RUN
from scripts.spi_baseline_exploration import sha
from src.run_external_corpus import _array_sha256
from src.utils import slugify


def audit(count):
    manifest=json.loads((DATA/'manifest.json').read_text());assert sha(DATA/'observations.npz')==manifest['archive_sha256']
    order=None;records=[]
    with np.load(DATA/'observations.npz') as raw:
        for r in manifest['rows'][:count]:
            folder=DATA/'mpis'/RUN/f"{r['corpus_index']+1:04d}-{slugify(r['row_id'],'dataset')}"
            meta=json.loads((folder/'meta.json').read_text())
            assert meta['status']=='complete' and meta['dataset_name']==r['row_id']
            assert (meta['M'],meta['T'])==(16,1000)
            assert meta['source']['archive_sha256']==manifest['archive_sha256']
            assert meta['source']['member_sha256']==_array_sha256(raw[r['row_id']])
            assert meta['experiment']['git_commit'].startswith('669a718') and not meta['experiment']['git_dirty']
            assert meta['random_seed']==261016 and meta['job']['estimator_rng_policy']=='isolated_serial'
            assert meta['pyspi']['version']['computation']=='3.0.0.r7'
            for key,path in [('corpus_config_sha256',f'configs/external/{RUN}.yaml'),('pyspi_config_sha256','configs/pyspi/benchmarked_p90.yaml'),('runner_sha256','src/run_external_corpus.py'),('compute_sha256','src/compute.py')]:
                assert meta['execution_identity'][key]==sha(ROOT/path)
            names=[s['name'] for s in meta['pyspi']['spis']]
            if order is None:order=names
            assert names==order and len(names)==289
            with np.load(folder/'spi_mpis.npz') as a:
                assert set(a.files)==set(names)
                off=~np.eye(16,dtype=bool)
                np.testing.assert_allclose(a['cov_EmpiricalCovariance'][off],np.cov(raw[r['row_id']],bias=True)[off],atol=1e-12,equal_nan=False)
                invalid=[k for k in names if not np.isfinite(a[k][off]).all()]
            records.append(dict(index=r['corpus_index']+1,row_id=r['row_id'],seconds=meta['job']['compute_seconds'],incomplete=invalid,
                mpi_sha256=sha(folder/'spi_mpis.npz'),meta_sha256=sha(folder/'meta.json')))
    result=dict(records=records,count=len(records),finite_mean_range=[min(289-len(r['incomplete']) for r in records),max(289-len(r['incomplete']) for r in records)],
        seconds_quantiles=np.quantile([r['seconds'] for r in records],[0,.5,.95,1]).tolist())
    (OUT/f'audit-{count}.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='records'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,required=True);a=p.parse_args();audit(a.count)
