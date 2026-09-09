"""Reuse audited MPI outputs to export normalized edges; never rerun pyspi."""
import argparse,json
from pathlib import Path
import numpy as np
from src.spi_edge_pool import standardize_edges
from src.representation_state_data import file_hash
from src.run_external_corpus import _atomic_json,_atomic_savez
from src.utils import slugify


def main(data,mpi_root,bank,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((data/'manifest.json').read_text());meta=json.loads(bank.with_suffix('.json').read_text())
    assert meta['artifact_sha256']==file_hash(bank)
    assert meta['manifest_sha256']==file_hash(data/'manifest.json')
    with np.load(bank,allow_pickle=False) as a:order=a['spi_order'].tolist();z=a['X_z']
    rows=manifest['rows'];lengths=np.array([r['M']*(r['M']-1) for r in rows]);k=len(order)
    edges=np.zeros((len(rows),max(lengths),k),dtype=np.float32);validity=np.zeros((len(rows),k),bool)
    maximum=0.;sources=[];upper=np.triu_indices(k,1)
    for i,(row,source) in enumerate(zip(rows,meta['sources'],strict=True)):
        assert row['row_id']==source['row_id']
        p=mpi_root/f"{row['corpus_index']:04d}-{slugify(row['row_id'],'dataset')}"/'spi_mpis.npz'
        assert file_hash(p)==source['mpi_sha256']
        with np.load(p,allow_pickle=False) as a:values,valid=standardize_edges(a,order)
        edges[i,:lengths[i]]=values;validity[i]=valid
        gram=values.astype(np.float64).T@values/len(values);gram[~valid,:]=np.nan;gram[:,~valid]=np.nan
        replay=gram[upper];np.testing.assert_allclose(replay,z[i],atol=3e-6,rtol=0,equal_nan=True)
        maximum=max(maximum,float(np.nanmax(abs(replay-z[i]))));sources.append(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    _atomic_savez(output,dict(edges=edges,validity=validity,lengths=lengths,spi_order=np.array(order),
        row_id=np.array([r['row_id'] for r in rows]),manifest_sha256=np.array(file_hash(data/'manifest.json'))))
    _atomic_json(output.with_suffix('.json'),dict(artifact_sha256=file_hash(output),base_bank_sha256=file_hash(bank),
        manifest_sha256=file_hash(data/'manifest.json'),rows=len(rows),z_recovery_max_difference=maximum,
        sources=sources,code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/spi_edge_pool.py']}))
    print(dict(rows=len(rows),z_recovery_max_difference=maximum,sha256=file_hash(output)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','mpi-root','bank','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();main(a.data,a.mpi_root,a.bank,a.output)
