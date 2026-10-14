"""Transfer the fixed 31.25-ms character coordinate; assess frozen N1 support."""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .acquire import sha256
from .analyse import EXPECTED_SOURCE
from .interpret import agreement, circular_lag, delete_range
from .prepare import lag_matrix
from .plots import style, save
from .responses import match

CONFIG=Path(__file__).parent/'configs'


def active_runs(pilot,cohort):
    """Resolve only active archives, never the quarantined pilot73 or duplicate56."""
    pilot_selection=json.loads((CONFIG/'selection.json').read_text())['records']
    pilot_corpora=json.loads((CONFIG/'analysis-selection.json').read_text())['corpora']
    scope={r['run_id']:r for r in json.loads((cohort/'scope/runs.json').read_text())}
    selected=[]
    for root,folders in [(pilot,pilot_corpora),(cohort,[Path(p).parent.name for p in json.loads((cohort/'active-configs.json').read_text())])]:
        for name in folders:
            folder=root/'corpora'/name
            manifest=json.loads((folder/'manifest.json').read_text())
            subject=manifest['rows'][0]['subject']
            if root==pilot:
                item=next(r for r in pilot_selection if r['subject']==subject)
                record=dict(item,run_id=Path(item['path']).name.removesuffix('_ieeg.json'),channels=manifest['rows'][0]['channels'])
            else:record=scope[name]
            selected.append((folder,manifest,record))
    assert len(selected)==34 and len({r['run_id'] for _,_,r in selected})==34
    assert len({r['subject'] for _,_,r in selected})==24
    return selected


def axis_features(runs,out):
    rows=[];provenance=[];audits={}
    for folder,_,_ in runs:
        root=folder.parent.parent
        if root not in audits:
            report=json.loads((root/'audit-latest.json').read_text())
            assert report['counts'].get('invalid',0)==0
            audits[root]={r['name']:r for r in report['rows']}
    pilot_subjects={r['subject'] for r in json.loads((CONFIG/'selection.json').read_text())['records']}
    for folder,manifest,record in runs:
        archive=folder/'observations.npz'
        assert sha256(archive)==manifest['archive_sha256']
        outputs={}
        for path in folder.glob('mpis/**/meta.json'):
            meta=json.loads(path.read_text());outputs[meta['source']['member']]=(path,meta)
        assert set(outputs)=={r['name'] for r in manifest['rows']}
        with np.load(archive) as bank:
            for row in manifest['rows']:
                x=bank[row['name']];path,meta=outputs[row['name']]
                assert meta['status']=='complete' and meta['source']['archive_sha256']==manifest['archive_sha256']
                assert meta['pyspi']['version']['computation']=='3.0.0.r9'
                assert EXPECTED_SOURCE in meta['pyspi']['version']['source']
                assert not meta['pyspi'].get('sampling') and x.shape==(row['M'],4096)
                edge=np.triu_indices(len(x),1)
                zero=np.corrcoef(x);lag=lag_matrix(x,4/128,128)
                with np.load(path.parent/'spi_mpis.npz') as table:
                    assert len(table.files)==289
                    assert set(table.files)=={r['name'] for r in meta['pyspi']['spis']}
                    mask=~np.eye(len(x),dtype=bool)
                    coverage=float(np.mean([np.isfinite(table[k][mask]).mean() for k in table.files]))
                    previous=audits[folder.parent.parent][row['name']]
                    assert previous['status']=='valid' and coverage>=.9
                    assert abs(coverage-previous['finite_edge_fraction'])<1e-12
                    np.testing.assert_allclose(zero[edge],table['cov_EmpiricalCovariance'][edge],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(lag[edge],table['corr_pearson_tau-4'][edge],atol=1e-12,rtol=1e-12)
                replay,fraction,removed=circular_lag(x,4)
                lo,hi=delete_range(zero,lag)
                rows.append(dict(name=row['name'],subject=row['subject'],run=record['run_id'],window=row['window'],
                    start=row['start'],reference_control=row['reference_control'],M=len(x),new_participant=row['subject'] not in pilot_subjects,
                    z=agreement(zero,lag),band_fraction=fraction,z_replay=agreement(zero,replay),
                    replay_edge_rmse=float(np.sqrt(np.mean((replay[edge]-lag[edge])**2))),
                    z_ablated=agreement(np.corrcoef(removed),lag_matrix(removed,4/128,128)),delete_min=lo,delete_max=hi,
                    mean_zero=float(zero[edge].mean()),mean_lag=float(lag[edge].mean())))
                provenance.append(dict(name=row['name'],archive_sha256=manifest['archive_sha256'],
                    mpi_sha256=sha256(path.parent/'spi_mpis.npz'),meta_sha256=sha256(path),finite_edge_fraction=coverage))
    frame=pd.DataFrame(rows);assert len(frame)==102
    frame.to_csv(out/'character-axis.csv',index=False)
    (out/'input-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return frame


def participant_medians(frame,columns):
    """Equal-weight runs within each participant; windows are not replicates."""
    return frame.groupby(['subject','run'])[columns].median().groupby('subject').median()


def witness(vectors,subjects,required,participants=4):
    """Exact support search with upper-bound pruning, at most one site/person."""
    vectors=np.asarray(vectors,dtype=int);subjects=np.asarray(subjects)
    if len(vectors)==0:return None
    def extend(chosen,common,candidates):
        need=participants-len(chosen)
        if need==0:return chosen
        if len(set(subjects[candidates]))<need:return None
        for j,index in enumerate(candidates):
            current=vectors[index] if common is None else np.minimum(common,vectors[index])
            if current.sum()<required:continue
            if need==1:return chosen+[index]
            rest=candidates[j+1:]
            rest=rest[subjects[rest]!=subjects[index]]
            if len(rest):rest=rest[np.minimum(vectors[rest],current).sum(axis=1)>=required]
            result=extend(chosen+[index],current,rest)
            if result is not None:return result
        return None
    return extend([],None,np.arange(len(vectors)))


def n1_support(runs,pilot,cohort,out):
    selected=[dict(record,fs=record.get('native_fs',record.get('fs'))) for _,_,record in runs]
    # Reuse the frozen event/site exclusions and native tt lookup, now for34 exact runs.
    responses_root=out/'n1';responses_root.mkdir(exist_ok=True)
    metadata=responses_root/'metadata';metadata.mkdir(exist_ok=True)
    provenance={}
    tree=json.loads((cohort/'tree.json').read_text())
    blobs={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    def verify_blob(path,relative):
        data=path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blobs[relative]
    for folder,manifest,record in runs:
        source=pilot if folder.is_relative_to(pilot) else cohort
        for suffix in ['channels.tsv','events.tsv']:
            relative=record['path'].replace('ieeg.json',suffix)
            path=source/'metadata'/relative;dest=metadata/relative
            verify_blob(path,relative)
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(path.read_bytes());provenance[relative]=sha256(path)
    match(responses_root,metadata,selected=selected,participants=[r for _,_,r in runs],
          derivative_path=pilot/'response-reference/raw/derivatives/av_ccep/ccepData_V2.mat')
    responses=pd.read_csv(responses_root/'responses/matched-latencies.csv')
    assert not responses.duplicated(['subject','run','site','response']).any()
    sites=[];strata=set();missing_contacts=[];missing_stim_sites=[];contact_maps={};electrode_maps={}
    for who,group in responses[responses.detected].groupby('subject'):
        paths=list((cohort/'metadata'/who).glob('**/*electrodes.tsv'))
        if not paths:paths=list((pilot/'metadata'/who).glob('**/*electrodes.tsv'))
        assert len(paths)==1,(who,paths)
        path=paths[0];relative=str(path.relative_to(cohort/'metadata' if path.is_relative_to(cohort/'metadata') else pilot/'metadata'))
        verify_blob(path,relative);provenance[str(path)]=sha256(path)
        electrodes=pd.read_csv(path,sep='\t').set_index('name');electrode_maps[who]=electrodes
        for (run,site),receivers in group.groupby(['run','site']):
            if len(receivers)<8:continue
            stim=site.split('-')[:2]
            if not set(stim)<=set(electrodes.index):
                missing_stim_sites.append(dict(subject=who,run=run,site=site));continue
            centre=electrodes.loc[stim,['x','y','z']].to_numpy(float).mean(axis=0)
            counts=Counter();contacts={}
            if not np.isfinite(centre).all():continue
            for response in receivers.response:
                if response not in electrodes.index:
                    missing_contacts.append(dict(subject=who,run=run,site=site,response=response));continue
                e=electrodes.loc[response];xyz=e[['x','y','z']].to_numpy(float)
                if not np.isfinite(xyz).all() or pd.isna(e.Destrieux_label_text):continue
                key=(e.Destrieux_label_text,int(np.linalg.norm(xyz-centre)//20))
                counts[key]+=1;contacts.setdefault(key,[]).append(response)
            strata.update(counts);sites.append(dict(subject=who,run=run,site=site,counts=counts))
            contact_maps[(who,run,site)]=contacts
    keys=sorted(strata);vectors=np.array([[s['counts'][k] for k in keys] for s in sites],dtype=int)
    subjects=[s['subject'] for s in sites]
    # Maximum achievable capacity across any four distinct participants, exact not random search.
    low=0;high=int(vectors.sum(axis=1).max()) if len(vectors) else 0;best=None
    while low<high:
        middle=(low+high+1)//2;found=witness(vectors,subjects,middle)
        if found is None:high=middle-1
        else:low=middle;best=found
    if low:best=witness(vectors,subjects,low)
    chosen=[{k:v for k,v in sites[i].items() if k!='counts'} for i in best] if best else []
    max_people=0;feasible_witness=None
    for count in range(4,len(set(subjects))+1):
        found=witness(vectors,subjects,8,participants=count)
        if found is None:break
        max_people=count;feasible_witness=found
    maximum_capacity_witness=chosen
    chosen=[{k:v for k,v in sites[i].items() if k!='counts'} for i in feasible_witness] if feasible_witness else []
    geometry=[];common=[]
    if feasible_witness:
        capacity=np.min(vectors[feasible_witness],axis=0)
        common=[dict(label=k[0],distance_bin_mm=[20*k[1],20*(k[1]+1)],count=int(n)) for k,n in zip(keys,capacity) if n]
        rng=np.random.default_rng(261014)
        slots=[k for k,n in zip(keys,capacity) for _ in range(n)]
        for repeat in range(256):
            selected_slots=Counter(slots[i] for i in rng.choice(len(slots),8,replace=False))
            for item in chosen:
                who,run,site=[item[k] for k in ['subject','run','site']]
                contacts=contact_maps[(who,run,site)];picked=[]
                for key,n in selected_slots.items():picked.extend(rng.choice(contacts[key],n,replace=False).tolist())
                xyz=electrode_maps[who].loc[picked,['x','y','z']].to_numpy(float)
                distances=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=2)[np.triu_indices(8,1)]
                geometry.append(dict(subject=who,run=run,site=site,resample=repeat,contacts=','.join(picked),
                    pair_distance_median_mm=float(np.median(distances)),pair_distance_q25_mm=float(np.quantile(distances,.25)),
                    pair_distance_q75_mm=float(np.quantile(distances,.75))))
        pd.DataFrame(geometry).to_csv(out/'n1-matched-geometry.csv',index=False)
    summary=dict(participants_with_responses=int(responses.subject.nunique()),matched_runs=int(responses.run.nunique()),
        detected_responses=int(responses.detected.sum()),sites_with_eight=len(sites),
        participants_with_eight=len(set(subjects)),site_counts=dict(Counter(subjects)),maximum_common_receivers=low,
        required_receivers=8,required_participants=4,coverage_gate_passed=low>=8,witness=chosen,maximum_capacity_witness=maximum_capacity_witness,
        maximum_participants_with_eight=max_people,shared_strata=common,
        receiver_pair_geometry='256 seed 261014 anatomy/stimulation-distance matched resamples of 8 contacts; diagnostic only, not yet pair-distance matched.',
        matching='Unchanged exact Destrieux label × 20 mm stimulation-centre distance bins; one site/run per distinct participant. Upper bound before receiver-pair distance matching.',
        selection='Detection/contact coverage only; no latency magnitudes or baseline z used to select support.',
        missing_electrode_contacts=missing_contacts,missing_stimulation_sites=missing_stim_sites,metadata_sha256=provenance)
    (out/'n1-coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def summarise(frame,out):
    car=frame[~frame.reference_control];reference=frame[frame.reference_control]
    columns=['z','band_fraction','z_ablated','mean_zero','mean_lag']
    subjects=participant_medians(car,columns)
    new_subjects=sorted(frame.loc[frame.new_participant,'subject'].unique())
    subjects['new_participant']=subjects.index.isin(new_subjects)
    first=car.sort_values('window').groupby('run').first().reset_index()
    paired=first.merge(reference,on=['subject','run','window'],suffixes=('_car','_ref'),validate='one_to_one')
    assert len(paired)==34
    paired['delta_z']=paired.z_ref-paired.z_car
    paired['absolute_delta_z']=paired.delta_z.abs()
    paired.to_csv(out/'reference-pairs.csv',index=False)
    paired_subject=paired.groupby('subject')[['z_car','z_ref','delta_z','absolute_delta_z']].median()
    subjects=subjects.join(paired_subject)
    window_rows=[]
    for run,g in car.groupby('run'):
        if len(g)<2:continue
        g=g.sort_values('window')
        window_rows.append(dict(subject=g.subject.iloc[0],run=run,n=len(g),first=g.z.iloc[0],last=g.z.iloc[-1],
                                span=float(g.z.max()-g.z.min()),same_sign=bool((g.z>0).all() or (g.z<0).all())))
    windows=pd.DataFrame(window_rows);windows.to_csv(out/'window-sensitivity.csv',index=False)
    subjects.to_csv(out/'participant-summary.csv')
    stats={}
    for label,people in [('all',subjects.index.tolist()),('new',new_subjects)]:
        c=car[car.subject.isin(people)];s=subjects.loc[people];p=paired_subject.loc[people]
        w=windows[windows.subject.isin(people)]
        wsub=w.groupby('subject')[['first','last','span']].median()
        stats[label]=dict(participants=len(people),windows=len(c),negative_windows=int((c.z<0).sum()),
            negative_any_participants=int(c[c.z<0].subject.nunique()),negative_all_participants=int(c.groupby('subject').z.max().lt(0).sum()),
            negative_median_participants=int((s.z<0).sum()),
            median_ablation_change=float((s.z_ablated-s.z).median()),
            z_band_spearman=float(spearmanr(s.z,s.band_fraction).statistic),
            z_band_leave_one_out_range=[float(v) for v in [min(spearmanr(s.drop(w).z,s.drop(w).band_fraction).statistic for w in s.index),max(spearmanr(s.drop(w).z,s.drop(w).band_fraction).statistic for w in s.index)]],
            reference_spearman=float(spearmanr(p.z_car,p.z_ref).statistic),
            reference_sign_agreement=int(((p.z_car>0)==(p.z_ref>0)).sum()),
            median_absolute_reference_change=float(p.absolute_delta_z.median()),max_absolute_reference_change=float(p.absolute_delta_z.max()),
            repeated_window_participants=len(wsub),repeated_window_runs=len(w),
            window_spearman=float(spearmanr(wsub['first'],wsub['last']).statistic),
            median_within_run_span=float(wsub.span.median()),max_within_run_span=float(w.span.max()),
            same_sign_repeated_runs=int(w.same_sign.sum()))
    stats['max_replay_z_error']=float(abs(frame.z-frame.z_replay).max())
    stats['max_replay_edge_rmse']=float(frame.replay_edge_rmse.max())
    stats['single_channel_deletion_sign_failures']=int(((car.z>0)&(car.delete_min<=0)| (car.z<0)&(car.delete_max>=0)).sum())
    stats['n1_association_run']=False
    stats['aggregation']='Median windows per run then median runs per participant. Reference: matched first window per run then participant median. No window/edge-level significance tests.'
    stats['fixed_axis']='corr_edges(Pearson zero lag, symmetric Pearson lag 4 at 128 Hz); fixed 8–24 Hz ablation. No PCA refit or feature selection.'
    return stats,subjects,paired,windows


def plots(frame,subjects,paired,windows,gate,root,runs):
    plt=style();car=frame[~frame.reference_control];ref=frame[frame.reference_control]
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    order=subjects.sort_values('z').index.tolist()
    for j,who in enumerate(order):
        g=car[car.subject==who];r=ref[ref.subject==who]
        color='#26828e' if subjects.loc[who,'new_participant'] else '#888888'
        axes[0,0].scatter(np.full(len(g),j),g.z,s=12,color=color,alpha=.7)
        axes[0,0].scatter(np.full(len(r),j),r.z,s=19,marker='x',color='#b56b22',linewidths=.8)
    axes[0,0].set(xticks=range(len(order)),xticklabels=[s[-2:] for s in order],ylabel='Zero-lag × 31.25 ms z',xlabel='Participant (sorted by CAR median)',title='CAR windows ● · acquisition ×')
    axes[0,0].tick_params(axis='x',labelrotation=90,labelsize=6)
    for fresh,color,label in [(False,'#888888','Pilot (5)'),(True,'#26828e','New (19)')]:
        s=subjects[subjects.new_participant==fresh]
        axes[0,1].scatter(s.z_car,s.z_ref,s=23,color=color,label=label)
        axes[1,0].scatter(s.band_fraction,s.z,s=23,color=color)
        axes[1,1].scatter(s.z,s.z_ablated,s=23,color=color)
    axes[0,1].plot([-1,1],[-1,1],':',lw=.8,color='.6');axes[0,1].legend(fontsize=7)
    axes[0,1].set(xlabel='Matched CAR z',ylabel='Acquisition-reference z',title='Participant medians across matched runs')
    axes[1,0].set(xlabel='8–24 Hz power fraction',ylabel='CAR z',title='Equal-weight participant summaries')
    axes[1,1].plot([-1,1],[-1,1],':',lw=.8,color='.6')
    axes[1,1].set(xlabel='CAR z',ylabel='After fixed 8–24 Hz removal',title='Spectral explanation; not validation')
    for ax in axes.flat:ax.set_box_aspect(1);ax.axhline(0,color='.8',lw=.5)
    save(fig,root/'figures','cohort-character-transfer')
    fig,axes=plt.subplots(1,2,figsize=(7,3.5))
    w=windows.groupby('subject')[['first','last']].median()
    axes[0].scatter(w['first'],w['last'],s=18,color='#26828e');axes[0].plot([-1,1],[-1,1],':',lw=.8,color='.6')
    axes[0].set(xlabel='First retained CAR window z',ylabel='Last retained CAR window z',title='Repeated-window participant medians')
    counts=gate['site_counts'];axes[1].bar([s[-2:] for s in counts],list(counts.values()),color='#26828e')
    axes[1].set(xlabel='Participant',ylabel='Run/sites with ≥8 detected receivers',title=f"Shared capacity across 4: {gate['maximum_common_receivers']} /8 required")
    axes[1].tick_params(axis='x',labelrotation=90,labelsize=6)
    for ax in axes:ax.set_box_aspect(1)
    save(fig,root/'figures','cohort-window-n1-support')
    # Illustrate the lowest participant-median coordinate in the independent extension.
    who=subjects[subjects.new_participant].z.idxmin()
    folder,manifest,_=next(r for r in runs if r[2]['subject']==who)
    chosen=min((r for r in manifest['rows'] if not r['reference_control']),key=lambda r:r['window'])
    reference=next(r for r in manifest['rows'] if r['reference_control'])
    with np.load(folder/'observations.npz') as bank:x=bank[chosen['name']];ref_x=bank[reference['name']]
    _,_,removed=circular_lag(x,4);edge=np.triu_indices(len(x),1)
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.3))
    for ax,signals,label in zip(axes,[x,ref_x,removed],['CAR','Acquisition reference','CAR; remove 8–24 Hz']):
        zero=np.corrcoef(signals);lag=lag_matrix(signals,4/128,128)
        ax.scatter(zero[edge],lag[edge],s=3,alpha=.25,color='#26828e',rasterized=True)
        ax.set(xlabel='Pearson zero lag',ylabel='Pearson 31.25 ms lag',xlim=(-.8,1),ylim=(-.8,1),
               title=f'{who[-2:]} · {label}\nz={agreement(zero,lag):.3f}');ax.set_box_aspect(1)
    save(fig,root/'figures','cohort-transfer-edges')


def main(pilot,cohort):
    pilot,cohort=Path(pilot),Path(cohort);out=cohort/'interpretation';out.mkdir(exist_ok=True)
    runs=active_runs(pilot,cohort)
    frame=axis_features(runs,out)
    gate=n1_support(runs,pilot,cohort,out)
    summary,subjects,paired,windows=summarise(frame,out)
    summary.update(n1=gate,
                   selection_sha256={p.name:sha256(p) for p in [CONFIG/'analysis-selection.json',CONFIG/'cohort-selection.json',CONFIG/'cohort-review.json',cohort/'active-configs.json']},
                   audit_sha256={p.parent.name:sha256(p) for p in [pilot/'audit-latest.json',cohort/'audit-latest.json']},script_sha256=sha256(__file__),interpret_source_sha256=sha256(Path(__file__).with_name('interpret.py')),
                   response_source_sha256=sha256(Path(__file__).with_name('responses.py')))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plots(frame,subjects,paired,windows,gate,cohort,runs)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',type=Path,required=True);parser.add_argument('--cohort',type=Path,required=True)
    args=parser.parse_args();report=main(args.pilot,args.cohort)
    print(json.dumps({key:report[key] for key in ['all','new']},indent=2))
    print('Maximum participants with eight matched receivers:',report['n1']['maximum_participants_with_eight'])
