"""Audit and perturb cross-SPI alignment while preserving every MPI edge multiset."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from src.representation_state_data import load_state_data,file_hash
from src.representation_mechanism import permute_dyads
from src.spi_spi_contract import build_unified_feature_values
from src.run_external_corpus import _atomic_json,_atomic_savez
from src.utils import slugify


def build(config,data,mpi_root,bank_path,output):
    if output.exists():raise FileExistsError(output)
    manifest,_=load_state_data(data,config)
    meta=json.loads(bank_path.with_suffix('.json').read_text())
    assert meta['artifact_sha256']==file_hash(bank_path)
    assert meta['manifest_sha256']==file_hash(data/'manifest.json')
    with np.load(bank_path,allow_pickle=False) as a:
        order=a['spi_order'].tolist();original=a['X_z']
        common={key:a[key] for key in ['X_m','X_validity','row_id','manifest_sha256']}
    seeds=[503,509,521]; banks=[[] for _ in seeds]; start=time.perf_counter();maximum=0.
    for row,source,z in zip(manifest['rows'],meta['sources'],original,strict=True):
        assert row['row_id']==source['row_id']
        path=mpi_root/f"{row['corpus_index']:04d}-{slugify(row['row_id'],'dataset')}"/'spi_mpis.npz'
        assert file_hash(path)==source['mpi_sha256']
        with np.load(path,allow_pickle=False) as a:mpis={name:a[name] for name in order}
        observed=build_unified_feature_values(mpis,order)[0]
        np.testing.assert_allclose(observed,z,atol=1e-10,rtol=1e-10,equal_nan=True)
        shared=permute_dyads(mpis,order,np.random.default_rng([499,row['corpus_index']]),shared=True)
        replay=build_unified_feature_values(shared,order)[0]
        np.testing.assert_allclose(replay,z,atol=1e-8,rtol=1e-8,equal_nan=True)
        maximum=max(maximum,float(np.nanmax(abs(replay-z))))
        mask=~np.eye(row['M'],dtype=bool)
        sorted_original=np.sort(np.asarray([mpis[name][mask] for name in order]),axis=1)
        for si,seed in enumerate(seeds):
            moved=permute_dyads(mpis,order,np.random.default_rng([seed,row['corpus_index']]))
            np.testing.assert_array_equal(sorted_original,np.sort(np.asarray([moved[name][mask] for name in order]),axis=1))
            null=build_unified_feature_values(moved,order)[0]
            np.testing.assert_array_equal(np.isfinite(null),np.isfinite(z))
            banks[si].append(null)
        if len(banks[0])%100==0:print(f'Perturbed {len(banks[0])}/{len(original)}',flush=True)
    output.mkdir(parents=True)
    for seed,values in zip(seeds,banks):
        path=output/f'null-{seed}.npz'
        _atomic_savez(path,{**common,'X_z':np.asarray(values)})
    _atomic_json(output/'verification.json',dict(seeds=seeds,records=len(original),edge_multisets_and_z_validity_preserved=True,
        shared_permutation_max_difference=maximum,seconds=time.perf_counter()-start,base_bank_sha256=file_hash(bank_path),
        artifacts={p.name:file_hash(p) for p in output.glob('*.npz')},
        code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/representation_mechanism.py','src/spi_spi_contract.py']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data','mpi-root','bank','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.config,a.data,a.mpi_root,a.bank,a.output)
