"""Fresh, fixed-scope p90 test of the Gaussian-margin t-copula scout."""
import json
import numpy as np
import yaml
from scripts.diagnose_tail_copula_margins import recording,population,ARMS,ROOT
from scripts.spi_baseline_exploration import sha

RUN='t-copula-normal-margins-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/'+RUN


def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True);rows=[];raw={};arm=ARMS[1]
    for swaps in range(5):
        truth=population(arm,swaps)
        for seed in range(200,232):
            name=f'normal-margin-t-swap{swaps}-s{seed}'
            x=recording(arm,swaps,seed)
            assert x.shape==(16,1000) and np.isfinite(x).all()
            np.testing.assert_allclose(x.std(1),1,atol=1e-12)
            raw[name]=x
            rows.append(dict(row_id=name,corpus_index=len(rows),M=16,N=16,T=1000,seed=seed,
                instance=seed,block=seed,label=RUN,system=RUN,control=swaps/4,
                role='development' if seed<216 else 'evaluation',**truth))
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([RUN])]*len(rows)),__shapes__=np.array([[16,1000]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    scope=('Fresh seeds200-231 after the focused three-arm scout; fixed rho/nu/levels and Gaussian margins. '
        'No physical bifurcation; tail dependence is known copula construction truth. Population Pearson changes slightly, '
        'so it is measured rather than asserted equal. PC1 feature selection is target-blind; supervised full-mean '
        'readouts test information availability, not like-for-like unsupervised performance. No outcome-selected PCs/windows.')
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),
        generator_sha256=sha(ROOT/'scripts/diagnose_tail_copula_margins.py'),builder_sha256=sha(__file__),
        analysis_scope=scope,figure_title='Gaussian margins, Student-t copula: M=N=16, T=1000; 16 training / 16 held seeds',
        pairing='SeedSequence includes arm,swap,seed; records independent across control levels, seed labels are split groups.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',
        sha256=manifest['archive_sha256'],axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',
        pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261075)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    smoke={1,160};node=set(np.linspace(2,159,48,dtype=int));rest=set(range(1,161))-smoke-node
    assert len(node)==48 and len(rest)==110
    for name,indices in [('smoke',smoke),('node',node),('rest',rest)]:
        (DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in sorted(indices)))
    print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},indent=2))


if __name__=='__main__':prepare()
