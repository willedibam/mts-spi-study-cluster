"""Check complete-grid provenance and independently verify mirrored pilot values."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from mtsspi.utils import slugify
from research.proof.baselines.strength_grid import DATA, OUT, ROOT, PROTOCOL, sha


def audit():
    receipt=json.loads((OUT/'execution.json').read_text())
    assert sha(DATA/'manifest.json')==receipt['manifest_sha256']
    assert sha(PROTOCOL)==receipt['protocol_sha256']
    assert sha(ROOT/'research/proof/baselines/strength_grid.py')==receipt['source_sha256']
    manifest=json.loads((DATA/'manifest.json').read_text())
    expected_code={name:hashlib.sha256(subprocess.check_output(['git','show',receipt['compute_source_commit']+':src/mtsspi/'+name],cwd=ROOT)).hexdigest() for name in ['run_corpus.py','corpus.py','generated_corpus.py','compute.py']}
    for cell in manifest['cells']:
        assert sha(DATA/(cell['cell']+'.npz'))==cell['archive_sha256']
    with np.load(OUT/'features.npz') as f:
        names=f['spi_order'];means=f['mean'];z=f['z'];ids=f['row_id']
    np.testing.assert_array_equal(ids,[r['row_id'] for r in manifest['rows']])
    assert len(names)==289 and means.shape==(1350,289) and z.shape==(1350,41616)
    upper=np.triu_indices(len(names),1);errors=[];missing=0;checked=0
    for i,row in enumerate(manifest['rows']):
        folder=DATA/'mpis'/slugify(row['cell_name'],'external-corpus')/f'{row["local_index"]:04d}-{slugify(row["row_id"],"dataset")}'
        meta=json.loads((folder/'meta.json').read_text())
        assert meta['status']=='complete' and meta['dataset_name']==row['row_id']
        assert (meta['M'],meta['T'])==(row['M'],row['T'])
        assert meta['source']['archive_sha256']==row['archive_sha256'] and meta['source']['member']==row['row_id']
        assert meta['execution_identity']['code_sha256']==expected_code
        assert meta['execution_identity']['corpus_config_sha256']==sha(DATA/(row['cell']+'.yaml'))
        assert meta['pyspi']['version']['source']==receipt['pyspi_source']+'/pyspi'
        assert meta['pyspi']['version']['computation']=='3.0.0.r9'
        assert meta['pyspi']['config_sha256']=='bc4bafa16b4add8bb6283db490a9fa3d8dde4fde47093147d32579b989397965'
        assert meta['random_seed']==2610152027 and not meta['normalise']
        assert meta['job']['estimator_rng_policy']=='pyspi_fixed_seeds'
        assert all(v=='1' for v in meta['job']['thread_variables'].values())
        np.testing.assert_array_equal(names,[s['name'] for s in meta['pyspi']['spis']])
        assert not any(any(token in message for token in ['ImportError:','ModuleNotFoundError:','AttributeError:','NameError:','TypeError:','KeyError:']) for message in meta['pyspi'].get('errors',{}).values())
        # Only the 16 input-selected early pilots are mirrored with numeric MPIs.
        path=folder/'spi_mpis.npz'
        if not path.exists():continue
        with np.load(path) as a:raw=np.array([a[n] for n in names])
        off=raw[:,~np.eye(row['M'],dtype=bool)]
        np.testing.assert_allclose(off.mean(1),means[i],rtol=1e-10,atol=1e-10,equal_nan=True)
        vectors=((raw+raw.transpose(0,2,1))/2)[:,np.triu_indices(row['M'],1)[0],np.triu_indices(row['M'],1)[1]]
        for k in np.linspace(0,len(upper[0])-1,100,dtype=int):
            u,v=vectors[upper[0][k]],vectors[upper[1][k]]
            if not (np.isfinite(u).all() and np.isfinite(v).all()) or np.linalg.norm(u-u.mean())<1e-12 or np.linalg.norm(v-v.mean())<1e-12:
                assert not np.isfinite(z[i,k]);missing+=1
            else:
                error=abs(np.corrcoef(u,v)[0,1]-z[i,k]);errors.append(error);assert error<1e-6
        checked+=1
    assert checked==16
    result=dict(records=1350,code_sha256=expected_code,independent_records=checked,mean_checks=checked*289,z_checks=checked*100,undefined_z_checks=missing,max_z_error=float(max(errors)),features_sha256=sha(OUT/'features.npz'),source_sha256=sha(__file__))
    (OUT/'final-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':audit()
