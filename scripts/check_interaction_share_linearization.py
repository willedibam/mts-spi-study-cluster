"""How much of the target survives replacing the true Jacobian by its time mean?"""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.check_interaction_share_references import simulate
from scripts.check_interaction_share_feasibility import interaction_share
from src.representation_state_data import load_state_data, file_hash


def run(config,data,output):
    if output.exists(): raise FileExistsError(output)
    manifest,_=load_state_data(data,config)
    results=[]
    for i,record in enumerate(manifest['masters']):
        _,jac=simulate(record['family'],record['a'],record['b'],record['seed_parts']+[0])
        jac=jac[1000:]; mean=jac.mean(axis=0)
        q=interaction_share(jac); constant=interaction_share(mean[None])
        fraction=float(np.mean(np.sum((jac-mean)**2,axis=(1,2)))/np.mean(np.sum(jac**2,axis=(1,2))))
        np.testing.assert_allclose(q-constant,fraction*(1-q)/(1-fraction),atol=1e-12)
        saved=[r['target'] for r in manifest['rows'] if r['master_index']==i]
        assert all(q==x for x in saved)
        results.append(dict(master_id=record['master_id'],family=record['family'],q=q,
                            mean_jacobian_share=constant,target_gap=q-constant,variable_jacobian_energy_fraction=fraction))
    summary={}
    for family in ['linear','tanh']:
        rows=[r for r in results if r['family']==family]
        summary[family]={key:dict(mean=float(np.mean([r[key] for r in rows])),maximum=float(max(r[key] for r in rows)))
                         for key in ['target_gap','variable_jacobian_energy_fraction']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(summary=summary,records=results,manifest_sha256=file_hash(data/'manifest.json'),
                     code_sha256=file_hash(Path(__file__)),claim='Oracle diagnostic; not an observed-data predictor or proof that the best linear predictor equals the mean Jacobian.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.config,a.data,a.output)
