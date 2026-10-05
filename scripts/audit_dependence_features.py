"""Independent NumPy replay of scalar means, MPI means and ordered z values."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.utils import slugify
from scripts.spi_baseline_exploration import sha


def audit(data,output,corpus):
    rows=json.loads((data/'manifest.json').read_text())['rows']
    bank=np.load(output/'features.npz');order=bank['spi_order']
    indices=np.unique(np.linspace(0,len(rows)-1,8,dtype=int))
    maximum_mean_error=maximum_z_error=maximum_scalar_error=0.
    with np.load(data/'observations.npz') as raw:
        for i,row in enumerate(rows):
            x=raw[row['row_id']];mask=~np.eye(len(x),dtype=bool)
            correlation=np.corrcoef(x)[mask]
            if 'CLE' in row:
                error=max(abs(correlation.mean()-row['mean_r']),abs(np.abs(correlation).mean()-row['mean_abs_r']))
                maximum_scalar_error=max(maximum_scalar_error,error)
                assert error<1e-12
            if i not in indices:continue
            folder=data/'mpis'/corpus/f"{row['corpus_index']+1:04d}-{slugify(row['row_id'],'dataset')}"
            with np.load(folder/'spi_mpis.npz') as mpis:
                vectors=np.array([mpis[k][mask] for k in order])
            finite=np.isfinite(vectors).all(1)
            expected_mean=np.full(len(order),np.nan);expected_mean[finite]=vectors[finite].mean(1)
            np.testing.assert_allclose(expected_mean,bank['mean'][i],equal_nan=True,atol=1e-12)
            maximum_mean_error=max(maximum_mean_error,float(np.nanmax(abs(expected_mean-bank['mean'][i]))))
            centered=vectors-vectors.mean(1,keepdims=True)
            valid=finite&(np.linalg.norm(centered,axis=1)>=1e-12)
            expected_matrix=np.full((len(order),len(order)),np.nan)
            expected_matrix[np.ix_(valid,valid)]=np.corrcoef(vectors[valid])
            expected_z=expected_matrix[np.triu_indices(len(order),1)]
            np.testing.assert_allclose(expected_z,bank['z'][i],equal_nan=True,atol=1e-6)
            maximum_z_error=max(maximum_z_error,float(np.nanmax(abs(expected_z-bank['z'][i]))))
    result=dict(records=len(rows),all_raw_scalar_means_checked='CLE' in rows[0],
                independently_replayed_mpi_rows=indices.tolist(),max_scalar_error=maximum_scalar_error,
                max_mean_error=maximum_mean_error,max_z_error=maximum_z_error,
                features_sha256=sha(output/'features.npz'),source_sha256=sha(__file__))
    (output/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus',required=True)
    a=p.parse_args();audit(a.data,a.output,a.corpus)
