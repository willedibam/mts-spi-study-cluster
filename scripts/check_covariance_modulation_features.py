"""Replay sampled MPI features and check raw fits after the PLS rank repair."""
import json
import tarfile
from pathlib import Path
import numpy as np
import yaml
from src.interaction_share_learning import fit_statistical,select_statistical
from src.mpi_representation_baselines import summarize_mpis
from src.representation_attribution import rich_marginals
from src.representation_state_data import load_state_data,observed_view,file_hash
from src.run_external_corpus import _array_sha256
from src.spi_spi_contract import build_unified_features


def main():
    root=Path('results/covariance_modulation_260909');data=Path('data/covariance_modulation_260909')
    config=Path('configs/analysis/covariance-modulation-260909.yaml');cfg=yaml.safe_load(config.read_text())
    manifest,masters=load_state_data(data,config);rows=manifest['rows']
    lookup={r['row_id']:i for i,r in enumerate(rows)}
    replay=root/'replay-mpis';replay.mkdir(exist_ok=True)
    with tarfile.open(root/'replay-mpis.tar.gz') as a:a.extractall(replay,filter='data')
    path=root/'gadi-analysis/features.npz'
    assert file_hash(path)==json.loads(path.with_suffix('.json').read_text())['artifact_sha256']
    checks=[]
    with np.load(path,allow_pickle=False) as bank:
        order=bank['spi_order'].tolist()
        matrices={k:bank['X_'+k] for k in ['m','g','z','validity']}
        for meta_path in sorted(replay.glob('*/meta.json')):
            meta=json.loads(meta_path.read_text());i=lookup[meta['dataset_name']];r=rows[i]
            view=observed_view(masters[r['master_index']],r['M'],r['T'])
            assert _array_sha256(view)==meta['source']['member_sha256']
            with np.load(meta_path.parent/'spi_mpis.npz',allow_pickle=False) as a:mpis={k:a[k] for k in order}
            _,g,valid=summarize_mpis(mpis,order)
            values={'m':rich_marginals(mpis,order),'g':g,'validity':valid,
                    'z':build_unified_features(mpis,order,metric='pearson').z}
            errors={}
            for key,value in values.items():
                np.testing.assert_allclose(value,matrices[key][i],atol=1e-10,rtol=1e-10,equal_nan=True)
                delta=np.abs(np.asarray(value,dtype=float)-matrices[key][i]);finite=delta[np.isfinite(delta)]
                errors[key]=float(finite.max()) if len(finite) else 0.
            checks.append(dict(row_id=r['row_id'],maximum_differences=errors))
    assert len(checks)==8
    with np.load(data/'raw-references.npz',allow_pickle=False) as a:raw={k:a[k] for k in a.files if k!='row_id'}
    y=np.array([r['target'] for r in rows]);strata=np.array([r['coupling_index'] for r in rows]);raw_checks=[]
    for p in sorted((root/'raw').glob('*/*.json')):
        info=json.loads(p.read_text());ident=info['identity'];name=ident['method']
        if not name.startswith('raw:'):continue
        keys=name[4:].rsplit('-',1)[0].split('+');bank={'u':np.concatenate([raw[k] for k in keys],axis=1)}
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
            train,evaluation,old=a['train_indices'],a['evaluation_indices'],a['prediction']
        chosen,details=select_statistical(bank,'u',train,y,strata,cfg['methods'],ident['seed'],'pls')
        assert list(chosen)==info['details']['chosen']
        cv_difference=max(abs(v-w) for new,previous in zip(details['candidates'],info['details']['candidates'],strict=True)
                          for v,w in zip(new['MAE'],previous['MAE'],strict=True))
        transform,model=fit_statistical(bank,'u',train,y,cfg['methods']['preprocessing'],'pls',*chosen)
        prediction=np.clip(model.predict(transform.transform(bank,evaluation)).reshape(-1),0,1)
        np.testing.assert_allclose(prediction,old,atol=1e-10,rtol=0)
        assert cv_difference<1e-10
        raw_checks.append(dict(fit=str(p),maximum_prediction_difference=float(abs(prediction-old).max()),
                               maximum_CV_difference=cv_difference))
    assert len(raw_checks)==108
    result=dict(status='passed',MPI_replays=checks,raw_rank_fix_replays=raw_checks,
                checker_sha256=file_hash(Path(__file__)))
    (root/'feature-and-raw-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(MPI_replays=len(checks),raw_fit_replays=len(raw_checks),
               max_raw_prediction_difference=max(r['maximum_prediction_difference'] for r in raw_checks)))


if __name__=='__main__':main()
