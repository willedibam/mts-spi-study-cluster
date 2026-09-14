"""Physics-only size, preparation and orbit-stability diagnosis of the 2D CML."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from numba import njit
import pandas as pd

from scripts.scout_cml2d_period_doubling import evolve, order_summary, step

SIZES = (5,6,7,8,9,10,12,16,24,32,64)
CONTROLS = (3.84,3.858,3.86212,3.866,3.89)
ANCHORS = (3.84,3.86212,3.89)
SEEDS = tuple(range(260914201,260914209))


def structural_cases():
    cases = [dict(L=L,r=r,seed=seed,preparation='random',burn=200000)
        for L,r,seed in itertools.product(SIZES,CONTROLS,SEEDS)]
    for L,r,seed in itertools.product((6,8),ANCHORS,SEEDS):
        for name,pre_r in [('low_prepared',3.84),('high_prepared',3.89)]:
            cases.append(dict(L=L,r=r,seed=seed,preparation=name,burn=200000,pre_r=pre_r))
        cases.append(dict(L=L,r=r,seed=seed,preparation='long_burn',burn=2000000))
    original_controls = (3.84,3.845,3.85,3.854,3.858,3.86006,3.86212,3.86306,
        3.864,3.865,3.866,3.868,3.87,3.8725,3.875,3.8825,3.89)
    original_seeds = tuple(range(260914001,260914009))+tuple(range(260914101,260914133))
    for L,r,seed,amplitude in itertools.product((6,8),ANCHORS,(260914001,260914002,260914101),(1e-12,1e-8)):
        index = (0 if L==6 else 680)+40*original_controls.index(r)+original_seeds.index(seed)
        cases.append(dict(L=L,r=r,seed=seed,preparation=f'perturb_{amplitude:g}',
            burn=200000,perturbation=amplitude,original_index=index))
    return cases


def recurrence_period(x, tolerance=1e-10):
    for period in range(1,33):
        if np.max(np.abs(x[period:]-x[:-period])) <= tolerance:
            return period
    return 0  # no recurrence detected, not a chaos classification


def coupling_matrix(side,g):
    n=side*side
    matrix=np.eye(n)*(1-4*g)
    for i,j in itertools.product(range(side),repeat=2):
        row=i*side+j
        for a,b in (((i-1)%side,j),((i+1)%side,j),(i,(j-1)%side),(i,(j+1)%side)):
            matrix[row,a*side+b] += g
    return matrix


def orbit_diagnostics(observed,r,g=.2):
    x=np.asarray(observed[-500:],dtype=float)
    side=int(np.sqrt(x.shape[1]))
    assert side**2==x.shape[1] and len(x)>64
    period=recurrence_period(x)
    result=dict(micro_period=period,global_period=recurrence_period(x.mean(axis=1)),
        constant_channels=int((x.std(axis=0)<=1e-8).sum()),
        minimum_channel_sd=float(x.std(axis=0).min()),
        mean_channel_sd=float(x.std(axis=0).mean()),
        floquet_radius=None,floquet_log_per_step=None)
    if period and side**2<=144:
        matrix=coupling_matrix(side,g)
        product=np.eye(side**2)
        cycle=x[-period:]
        for state in cycle:
            product=(matrix*(r*(1-2*state))[None,:])@product
        radius=float(np.max(np.abs(np.linalg.eigvals(product))))
        result.update(floquet_radius=radius,
            floquet_log_per_step=float(np.log(radius)/period) if radius>0 else None)
    fields=x[-64:].reshape(-1,side,side)
    for axis,name in [(1,'row'),(2,'column')]:
        translations=[shift for shift in range(1,side)
            if np.max(np.abs(fields-np.roll(fields,shift,axis=axis)))<=1e-10]
        result[f'spatial_{name}_period']=translations[0] if translations else side
    return result


@njit(cache=True)
def tangent_step(state,tangent,r,g):
    side=state.shape[0]
    mapped=r*(1-2*state)*tangent
    output=np.empty_like(tangent)
    for i in range(side):
        for j in range(side):
            output[i,j]=(1-4*g)*mapped[i,j]+g*(mapped[(i-1)%side,j]+
                mapped[(i+1)%side,j]+mapped[i,(j-1)%side]+mapped[i,(j+1)%side])
    return output


@njit(cache=True)
def lyapunov_blocks(state,tangent,r,g,alignment=5000,steps=100000):
    state=state.copy()
    tangent=tangent.copy()/np.sqrt(np.sum(tangent*tangent))
    mapped=np.empty_like(state);output=np.empty_like(state)
    blocks=np.zeros(10)
    for t in range(alignment+steps):
        next_tangent=tangent_step(state,tangent,r,g)
        norm=np.sqrt(np.sum(next_tangent*next_tangent))
        if norm==0:
            return np.full(10,-np.inf)
        tangent=next_tangent/norm
        step(state,mapped,output,r,g)
        state,output=output,state
        if t>=alignment:
            blocks[(t-alignment)//(steps//10)]+=np.log(norm)/(steps//10)
    return blocks


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def audit_case(physics,output,index):
    target=output/f'audit-{index:04d}.json'
    if target.exists():raise FileExistsError(target)
    path=physics/f'case-{index:03d}.npz'
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['metadata_json']))
        assert meta['views']==['full']
        observed=a['observed'][:,0]
        result={key:meta[key] for key in ('L','r','seed','Q','Q_first_half','Q_second_half')}
        result.update(orbit_diagnostics(observed,meta['r']),
            observation_sha256=hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest(),
            input_metadata_sha256=hashlib.sha256(str(a['metadata_json']).encode()).hexdigest())
    result.update(case_index=index,source_sha256=source_hash(),input_path=str(path))
    output.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')


def scout_case(physics,output,index):
    case=structural_cases()[index]
    target=output/f'scout-{index:04d}.npz'
    if target.exists():
        with np.load(target,allow_pickle=False) as a:
            saved=json.loads(str(a['metadata_json']))
            assert saved['source_sha256']==source_hash() and saved['case_index']==index
            assert all(saved[key]==value for key,value in case.items())
            assert np.isfinite(a['final_state']).all()
        return
    L,r=case['L'],case['r'];g=.2
    seeds=np.random.SeedSequence(case['seed']).generate_state(3)
    rng=np.random.default_rng(int(seeds[0]))
    ids=np.arange(L*L).reshape(1,-1)
    if 'original_index' in case:
        with np.load(physics/f"case-{case['original_index']:03d}.npz",allow_pickle=False) as a:
            state=a['final_state'].copy()
            original=json.loads(str(a['metadata_json']))
            assert original['L']==L and original['r']==r and original['seed']==case['seed']
        direction=rng.normal(size=(L,L));direction-=direction.mean()
        direction/=np.max(np.abs(direction))
        state=state+case['perturbation']*direction
        assert ((state>=0)&(state<=1)).all()
    else:
        state=rng.random((L,L))
    if 'pre_r' in case:
        _,_,state=evolve(state,case['pre_r'],g,200000,0,0,ids)
    means,observed,final=evolve(state,r,g,case['burn'],1002000,2000,ids)
    assert np.isfinite(means).all() and np.isfinite(observed).all()
    result={**case,**order_summary(means,2000),**orbit_diagnostics(observed[:,0],r)}
    tangent=np.random.default_rng(int(seeds[2])).normal(size=(L,L))
    blocks=lyapunov_blocks(final,tangent,r,g)
    result.update(lyapunov_mean=float(blocks.mean()),lyapunov_blocks=blocks.tolist(),
        N=L*L,g=g,record_steps=1002000,case_index=index,source_sha256=source_hash(),
        generator_sha256=hashlib.sha256(Path(__file__).with_name('scout_cml2d_period_doubling.py').read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as handle:
        np.savez_compressed(handle,metadata_json=json.dumps(result),
            initial_global_mean=means[:1000],late_global_mean=means[-8192:],
            initial_field=observed[:256,0],final_state=final)


def aggregate(root):
    audit=[json.loads(p.read_text()) for p in sorted((root/'audit').glob('audit-*.json'))]
    scout=[]
    for path in sorted((root/'scout').glob('scout-*.npz')):
        with np.load(path,allow_pickle=False) as a:
            scout.append(json.loads(str(a['metadata_json'])))
    assert len(audit)==1360 and len(scout)==620
    assert {r['case_index'] for r in audit}==set(range(1360))
    assert {r['case_index'] for r in scout}==set(range(620))
    pd.DataFrame(audit).to_csv(root/'existing-bank-audit.csv',index=False)
    pd.DataFrame(scout).to_csv(root/'structural-scout.csv',index=False)
    (root/'complete.json').write_text(json.dumps(dict(audited=1360,new_cases=620,
        source_sha256=source_hash(),note='Finite-size/time and stability diagnostics; no pyspi and no universal minimum-size claim.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','scout','aggregate'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--physics',type=Path,required=True)
    p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.stage=='aggregate':aggregate(a.root)
    elif a.stage=='audit':audit_case(a.physics,a.root/'audit',a.index)
    else:scout_case(a.physics,a.root/'scout',a.index)
