"""Analytically selected binary band swap; no SPI outcomes select the contrast."""
import json
from pathlib import Path
import numpy as np
import yaml
from scripts.band_organization_experiment import simulate,direct_features,covariance_layers
from scripts.spi_baseline_exploration import ROOT,sha

DATA=ROOT/'data/representation/band-swap-261004'
OUT=ROOT/'results/representation/band-swap-261004'
CLASSES=('BCA','CBA')


def build():
    if (DATA/'manifest.json').exists():raise FileExistsError('Immutable bank already exists')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    matrices=covariance_layers(.55);bca=matrices[[1,2,0]];cba=matrices[[2,1,0]]
    np.testing.assert_allclose(bca.sum(axis=0),cba.sum(axis=0))
    np.testing.assert_allclose(bca[:2].sum(axis=0),cba[:2].sum(axis=0))
    np.testing.assert_allclose(np.maximum(bca[0],bca[1]),np.maximum(cba[0],cba[1]))
    arrays,rows,features={},{},{}
    rows=[]
    for block in range(64):
        for label in CLASSES:
            x,meta=simulate(label,block);name=f'{label}-block-{block:02d}';arrays[name]=x.T
            rows.append(dict(row_id=name,label=label,block=block,role='development' if block<32 else 'evaluation',
                development_part='train' if block<24 else 'validation' if block<32 else 'held',
                corpus_index=len(rows),M=16,T=1000,**meta))
            for key,value in direct_features(x).items():features.setdefault(key,[]).append(value)
    np.savez_compressed(DATA/'observations.npz',**arrays,
        __dataset_names__=np.array([r['row_id'] for r in rows]),__labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
        __shapes__=np.array([[16,1000]]*len(rows)),__axis_order__=np.array(['process','observation']))
    np.savez_compressed(DATA/'direct-features.npz',row_id=np.array([r['row_id'] for r in rows]),**features)
    m=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),script_sha256=sha(Path(__file__)),
        generator_sha256=sha(ROOT/'scripts/band_organization_experiment.py'),protocol_sha256=sha(ROOT/'configs/analysis/band-swap-261004.yaml'),
        population_equalities=['each individual band MPI marginal law','zero-lag covariance matrix','low-half covariance sum and elementwise maximum','high-band covariance matrix'],
        qualification='Full289 marginal equality remains an empirical question; this is functional organization, not causal cross-frequency coupling.')
    (DATA/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    config=yaml.safe_load((ROOT/'configs/external/band-organization-261003.yaml').read_text())
    for key in ('name','base_output_dir'):config[key]=config[key].replace('band-organization-fixed-strength-261003','band-swap-261004')
    config['source']['archive']=config['source']['archive'].replace('band-organization-fixed-strength-261003','band-swap-261004')
    config['source']['sha256']=m['archive_sha256']
    (ROOT/'configs/external/band-swap-261004.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print('Generated',len(rows),'records; archive',m['archive_sha256'])

if __name__=='__main__':build()
