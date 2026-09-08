"""Run the frozen raw feasibility scout; no fitted predictive model or p90."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from src.oscillatory_coorganization import simulate,observed_controls
from src.phase_surrogates import pooled_autospectrum
from src.covariance_modulation import raw_references
from src.representation_state_data import file_hash


def main():
    data=Path('data/oscillatory_coorganization_scout_260909')
    out=Path('results/oscillatory_coorganization_scout_260909')
    if data.exists() or out.exists():raise FileExistsError('Do not overwrite or duplicate scout')
    data.mkdir();out.mkdir();rows=[];corpus={};raw={};records=[]
    for block in range(24):
        for aligned in [False,True]:
            x,meta=simulate(aligned,[260909121,block]);records.append(dict(block=block,aligned=aligned,**meta))
            for m,t in [(16,1000),(8,500)]:
                view=x[:t,:m];assert np.isfinite(view).all()
                name=f'b{block:02d}-a{int(aligned)}-M{m}-T{t}';corpus[name]=view
                features=observed_controls(view)
                features={k:v for k,v in features.items() if not k.endswith('_matrix')}
                features['spectrum']=pooled_autospectrum(view)
                features.update({'raw_'+k:v for k,v in raw_references(view).items() if k!='moment_proxy'})
                for key,value in features.items():raw.setdefault(key,[]).append(value)
                rows.append(dict(row_id=name,block=block,target=int(aligned),M=m,T=t,corpus_index=len(rows)+1))
    names=np.asarray([r['row_id'] for r in rows]);corpus.update(__dataset_names__=names,
       __labels_json__=np.asarray(['[]']*len(rows)),__shapes__=np.asarray([[r['T'],r['M']] for r in rows]),
       __axis_order__=np.asarray(['observation','process']))
    np.savez_compressed(data/'views.npz',**corpus)
    np.savez_compressed(data/'raw.npz',**{k:np.asarray(v) for k,v in raw.items()},row_id=names)
    (data/'manifest.json').write_text(json.dumps(dict(rows=rows,records=records,
       protocol_sha256=file_hash(Path('docs/oscillatory-coorganization-scout.md')),
       code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/oscillatory_coorganization.py']},
       artifacts={p:file_hash(data/p) for p in ['views.npz','raw.npz']}),indent=2)+'\n')
    checks=[]
    for m in [16,8]:
        ix=np.array([r['M']==m for r in rows]);target=np.array([r['target'] for r in rows])[ix]
        for name,col in [('direct_agreement',0),('direct_agreement',1),('phase_summary',0),('envelope_summary',0)]:
            value=np.asarray(raw[name])[ix,col];auc=float(roc_auc_score(target,value))
            checks.append(dict(M=m,feature=name,column=col,AUROC=auc,
                               crossed_mean=float(value[target==0].mean()),aligned_mean=float(value[target==1].mean())))
    gate=all(c['AUROC']>=.8 for c in checks if c['feature']=='direct_agreement' and c['column']==0)
    result=dict(status='raw_feasibility_only',p90_gate_passed=gate,checks=checks,
                independent_paired_blocks=24,views=96,script_sha256=file_hash(Path(__file__)))
    (out/'raw-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
