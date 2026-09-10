"""Audit and summarize one complete NeuroTycho p90 bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from src.mpi_representation_baselines import summarize_mpis
from src.representation_attribution import rich_marginals
from src.spi_edge_pool import standardize_edges
from src.spi_spi_contract import build_unified_features,schema_sha256
from src.utils import slugify


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.data/'manifest.json').read_text());rows=manifest['rows']
    arrays={k:[] for k in ['m','g','z','validity','edges','lengths']};order=None;sources=[]
    versions=set();maximum=max(r['M']*(r['M']-1) for r in rows)
    for row in rows:
        root=args.mpi_root/f'{row["corpus_index"]:04d}-{slugify(row["row_id"],"dataset")}'
        metadata=root/'meta.json';mpi=root/'spi_mpis.npz';meta=json.loads(metadata.read_text())
        assert meta['status']=='complete' and meta['normalise'] is False
        assert meta['dataset_name']==row['row_id'] and (meta['M'],meta['T'])==(row['M'],row['T'])
        assert meta['source']['archive_sha256']==manifest['archive_sha256']
        assert meta['source']['member_sha256']==row['array_sha256']
        assert meta['pyspi']['config_sha256']==manifest['catalogue_sha256']
        names=[v['name'] for v in meta['pyspi']['spis']]
        if order is None:order=names
        assert names==order and len(order)==289
        with np.load(mpi) as bank:
            assert bank.files==order
            matrices={k:bank[k] for k in order}
        _,graph,valid=summarize_mpis(matrices,order)
        features=build_unified_features(matrices,order,metric='pearson')
        edges,edge_valid=standardize_edges(matrices,order)
        np.testing.assert_array_equal(valid,edge_valid)
        padded=np.zeros((maximum,289),dtype=np.float32);padded[:len(edges)]=edges
        for key,value in zip(arrays,[rich_marginals(matrices,order),graph,features.z,valid,padded,len(edges)],strict=True):
            arrays[key].append(value)
        versions.add(json.dumps(meta['pyspi']['version'],sort_keys=True))
        sources.append(dict(row_id=row['row_id'],mpi_sha256=sha(mpi),meta_sha256=sha(metadata),
            valid_spis=int(valid.sum()),errors=meta['pyspi']['errors'],seconds=meta['job']['compute_seconds']))
        if len(sources)%32==0:print('Built',len(sources),'/',len(rows),flush=True)
    assert len(versions)==1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    labels=['AwakeEyesClosed','Anesthetized']
    np.savez_compressed(args.output,**{k:np.asarray(v) for k,v in arrays.items()},
        row_id=np.array([r['row_id'] for r in rows]),
        record_id=np.array([f'{r["archive"]}/{r["session"]}/{labels[r["target"]]}/{r["start"]}' for r in rows]),
        animal=np.array([r['animal'] for r in rows]),archive=np.array([r['archive'] for r in rows]),
        y=np.array([r['target'] for r in rows]),M=np.array([r['M'] for r in rows]),T=np.array([r['T'] for r in rows]),
        spi_order=np.array(order),schema_sha256=schema_sha256(features.schema))
    report=dict(sha256=sha(args.output),manifest_sha256=sha(args.data/'manifest.json'),phase=manifest['phase'],
        catalogue_sha256=manifest['catalogue_sha256'],versions=list(versions),sources=sources,
        modules={str(p):sha(p) for p in [Path(__file__),Path('src/mpi_representation_baselines.py'),
            Path('src/representation_attribution.py'),Path('src/spi_spi_contract.py'),Path('src/spi_edge_pool.py')]})
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(dict(rows=len(rows),valid_min=min(s['valid_spis'] for s in sources),sha256=report['sha256']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--mpi-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
