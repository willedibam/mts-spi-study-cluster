"""Export immutable nested observations from existing CML2D master archives."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(input_dir, output_dir, config_path, M_values, T_values, views, development_seeds, select_L=None, select_r=None, select_seeds=None):
    if output_dir.exists() or config_path.exists(): raise FileExistsError('refusing to overwrite corpus/config')
    arrays={}; rows=[]; sources=[]
    for path in sorted(input_dir.glob('case-*.npz')):
        with np.load(path,allow_pickle=False) as a:
            meta=json.loads(str(a['metadata_json'])); observed=a['observed']; means=a['global_mean']
            if select_L is not None and meta['L'] != select_L: continue
            if select_r is not None and meta['r'] not in select_r: continue
            if select_seeds is not None and meta['seed'] not in select_seeds: continue
            source_hash=file_hash(path)
            for view in views:
                v=meta['views'].index(view)
                for M in M_values:
                    for T in T_values:
                        if M>observed.shape[2] or T>len(observed) or T%2: raise ValueError('invalid requested view')
                        name=f"r{meta['r']:g}-s{meta['seed']}-{view}-m{M}-t{T}"
                        x=np.ascontiguousarray(observed[:T,v,:M].T)
                        if not np.isfinite(x).all() or np.any(x.std(axis=1)<=1e-8): raise ValueError('invalid raw view')
                        arrays[name]=x
                        rows.append(dict(row_id=name,corpus_index=len(rows)+1,L=meta['L'],N=meta['N'],r=meta['r'],
                            seed=meta['seed'],view=view,M=M,T=T,role='development' if meta['seed'] in development_seeds else 'evaluation',
                            master=path.name,master_sha256=source_hash,
                            Q_reference=meta['Q'],Q_window=float(np.abs(means[1:T:2]-means[:T:2]).mean()),
                            Q_blocks=meta['Q_blocks']))
            sources.append(dict(path=str(path),sha256=source_hash,metadata=meta))
    if not rows or len(arrays)!=len(rows): raise ValueError('empty or duplicate corpus')
    arrays.update(__dataset_names__=np.array(list(arrays)),
        __labels_json__=np.array([json.dumps(['cml2d','period-doubling',row['view']]) for row in rows]),
        __shapes__=np.array([[row['M'],row['T']] for row in rows]),
        __axis_order__=np.array(['process','observation']))
    output_dir.mkdir(parents=True)
    with (output_dir/'observations.npz').open('xb') as handle: np.savez_compressed(handle,**arrays)
    sha=file_hash(output_dir/'observations.npz')
    manifest=dict(rows=rows,source_archives=sources,archive_sha256=sha,
        exporter_sha256=file_hash(__file__),development_seeds=development_seeds)
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=output_dir.name,source=dict(format='named-npz-v1',archive=str(output_dir/'observations.npz'),sha256=sha,axis_order=['process','observation']),
        base_output_dir=str(output_dir/'mpi'),pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=260911)
    config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    print(json.dumps(dict(rows=len(rows),master_archives=len(sources),config=str(config_path),sha256=sha)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input-dir','output-dir','config']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--M',type=int,nargs='+',default=[32]);p.add_argument('--T',type=int,nargs='+',default=[1000])
    p.add_argument('--views',nargs='+',default=['dispersed']);p.add_argument('--development-seeds',type=int,nargs='+',required=True)
    p.add_argument('--select-L',type=int);p.add_argument('--select-r',type=float,nargs='+');p.add_argument('--select-seeds',type=int,nargs='+')
    a=p.parse_args();prepare(a.input_dir,a.output_dir,a.config,a.M,a.T,a.views,a.development_seeds,a.select_L,a.select_r,a.select_seeds)
