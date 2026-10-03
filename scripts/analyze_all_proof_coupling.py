"""Shared frozen readout applied to the expanded bank, preserving the pilot results."""
import argparse,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts import analyze_native_coupling as common
from scripts.calibrate_all_proof_coupling import DATA,OUT
from scripts.spi_baseline_exploration import ROOT,sha


def rows():return pd.DataFrame(json.loads((OUT/'combined-manifest.json').read_text())['rows'])


def extract():
    extra=OUT/'extra';extra.mkdir(exist_ok=True)
    common.extract(DATA,extra,corpus='proof-strength-all-261003')
    paths=[ROOT/'results/representation/native-gain-261003/features.npz',extra/'features.npz']
    banks=[]
    for path in paths:
        with np.load(path) as a:banks.append({k:a[k] for k in a.files})
    np.testing.assert_array_equal(banks[0]['spi_order'],banks[1]['spi_order'])
    features={k:np.concatenate([a[k] for a in banks]) for k in ['mean','distribution','z','validity','row_id']}
    np.testing.assert_array_equal(features['row_id'],rows().row_id)
    np.savez_compressed(OUT/'features.npz',**features,spi_order=banks[0]['spi_order'])
    manifest=json.loads((OUT/'combined-manifest.json').read_text())
    for bank,digest in manifest['source_manifests'].items():
        assert sha(ROOT/f'data/representation/{bank}/manifest.json')==digest
    (OUT/'sources.json').write_text(json.dumps(dict(feature_sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
        combined_manifest_sha256=sha(OUT/'combined-manifest.json'),features_sha256=sha(OUT/'features.npz')),indent=2)+'\n')


def raw_diagnostics():
    records=[]
    for bank,group in rows().groupby('source_bank',sort=False):
        with np.load(ROOT/f'data/representation/{bank}/observations.npz') as a:
            for _,row in group.iterrows():
                x=a[row.row_id];sv=np.linalg.svd(x,compute_uv=False);weights=sv**2/sum(sv**2)
                lag={f'median_lag{k}':float(np.median([np.corrcoef(y[:-k],y[k:])[0,1] for y in x])) for k in (1,2)}
                records.append(dict(row_id=row.row_id,label=row.label,block=row.block,
                    mean_abs_Pearson=row.mean_abs_Pearson,min_scaled_channel_sd=np.std(x,axis=1).min(),
                    effective_rank=np.exp(-np.sum(weights*np.log(weights+1e-300))),**lag))
    pd.DataFrame(records).to_csv(OUT/'raw-diagnostics.csv',index=False)


def per_class():
    pred=pd.read_csv(OUT/'predictions.csv');pred['correct']=pred.label==pred.predicted
    pred.groupby(['scope','label','method']).correct.mean().unstack('method').to_csv(OUT/'per-class.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','analyze']);args=p.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='extract':extract()
        else:common.analyze(rows(),OUT);per_class();raw_diagnostics()
