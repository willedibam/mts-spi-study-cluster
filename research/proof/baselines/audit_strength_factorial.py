import json
from pathlib import Path
import numpy as np
from research.proof.baselines.strength_factorial import DATA,OUT,sha

def audit():
    manifest=json.loads((DATA/'manifest.json').read_text())
    with np.load(OUT/'features.npz') as f:
        names=f['spi_order']; means=f['mean']; z=f['z']; ids=f['row_id']
    upper=np.triu_indices(len(names),1)
    errors=[]; missing=0; count=0
    # Use every representative pilot, selected from input noise before SPI outcomes.
    for p in sorted((DATA/'mpis'/'strength-factorial-261015').glob('*/spi_mpis.npz')):
        meta=json.loads(p.with_name('meta.json').read_text()); i=list(ids).index(meta['dataset_name'])
        with np.load(p) as a:
            raw=np.array([a[n] for n in names]); off=raw[:,~np.eye(16,dtype=bool)]
            expected=off.mean(1)
            np.testing.assert_allclose(expected,means[i],rtol=1e-10,atol=1e-10,equal_nan=True)
            vectors=((raw+raw.transpose(0,2,1))/2)[:,np.triu_indices(16,1)[0],np.triu_indices(16,1)[1]]
            for k in np.linspace(0,len(upper[0])-1,100,dtype=int):
                u,v=vectors[upper[0][k]],vectors[upper[1][k]]
                if not (np.isfinite(u).all() and np.isfinite(v).all()) or u.std()==0 or v.std()==0:
                    assert not np.isfinite(z[i,k]); missing+=1
                else:
                    value=np.corrcoef(u,v)[0,1]; error=abs(value-z[i,k]); errors.append(error)
                    assert error<1e-6,(meta['dataset_name'],k,error)
            count+=1
    assert count==16
    result=dict(pilot_records=count,independent_mean_checks=count*len(names),independent_z_checks=count*100,
                undefined_z_checks=missing,max_z_error=float(max(errors)),features_sha256=sha(OUT/'features.npz'))
    (OUT/'independent-value-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':audit()
