"""Match published N1 detections to pilot runs/contacts and inspect native examples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from research.empirical.ccep_pilot.acquire import sha256


def site_key(value):
    return tuple(sorted(str(value).split('-')[:2]))


def latency_ms(samples, time):
    samples=np.asarray(samples,dtype=float);out=np.full(samples.shape,np.nan)
    valid=np.isfinite(samples)&(samples>=1)&(samples<=len(time))
    out[valid]=np.asarray(time)[np.rint(samples[valid]).astype(int)-1]*1000
    return out


def match(root,metadata_root=None):
    root=Path(root);out=root/'responses';out.mkdir(exist_ok=True)
    metadata_root=Path(metadata_root) if metadata_root is not None else root/'raw'
    path=root/'response-reference/raw/derivatives/av_ccep/ccepData_V2.mat'
    expected=json.loads((Path(__file__).parent/'configs/response-reference.json').read_text())
    assert sha256(path)==next(f['sha256'] for f in expected['files'] if f['path'].endswith('ccepData_V2.mat'))
    bank=loadmat(path,simplify_cells=True)['ccepData']
    selected=json.loads((Path(__file__).parent/'configs/selection.json').read_text())['records']
    participants=json.loads((root/'scope/participants.json').read_text())
    summaries=[];edges=[];sites=[];examples=[]
    for item in selected:
        subject=item['subject'];record=next(r for r in participants if r['subject']==subject)
        parent=next(r for r in bank if r['id']==subject)
        runs=parent['run'] if isinstance(parent['run'],list) else [parent['run']]
        target=Path(item['path']).name.replace('_ieeg.json','_averageCCEPs.mat')
        run=next(r for r in runs if r['runName']==target)
        names=list(run['channel_names']);retained=set(record['channels'])
        channels=pd.read_csv(metadata_root/item['path'].replace('_ieeg.json','_channels.tsv'),sep='\t')
        assert names==channels.name.tolist()
        events=pd.read_csv(metadata_root/item['path'].replace('_ieeg.json','_events.tsv'),sep='\t')
        pulses=events[events.trial_type=='electrical_stimulation'].copy()
        pulses['site_key']=pulses.electrical_stimulation_site.map(site_key)
        bad=[]
        for e in events.itertuples():
            if e.trial_type in ('artefact','artifact','seizure'):
                margin=300 if e.trial_type=='seizure' else 1
                bad.append((e.onset-margin,e.onset+max(e.duration,0)+margin))
        lat=latency_ms(run['n1_peak_sample'],run['tt'])
        local=[];eligible_sites=[]
        for column,name in enumerate(np.atleast_1d(run['stimpair_names'])):
            key=site_key(name)
            trials=pulses.loc[pulses.site_key.map(lambda x:x==key)]
            reason=None
            if len(key)!=2 or not set(key)<=retained:reason='stimulating contact outside retained ECoG'
            elif len(trials)<5:reason='fewer than five matching pulses'
            elif any(onset-2<b and onset+3>a for onset in trials.onset for a,b in bad):reason='published average includes a pulse overlapping exclusion margins'
            sites.append(dict(subject=subject,site=str(name),pulses=len(trials),eligible=reason is None,reason=reason))
            if reason:continue
            eligible_sites.append(str(name))
            for channel,contact in enumerate(names):
                if contact not in retained or contact in key:continue
                value=float(lat[channel,column]);valid=np.isfinite(value) and 9<value<=100.5
                row=dict(subject=subject,run=target,site=str(name),response=contact,
                         latency_ms=value if valid else np.nan,detected=bool(valid),n_pulses=len(trials),
                         currents_A=json.dumps(sorted(trials.electrical_stimulation_current.dropna().unique().tolist())))
                edges.append(row)
                if valid:local.append(row)
        values=np.array([r['latency_ms'] for r in local])
        # Equal weight per stimulation site for the primary aggregate dispersion.
        site_iqrs=[float(np.subtract(*np.quantile([r['latency_ms'] for r in local if r['site']==site],[.75,.25])))
                   for site in eligible_sites if sum(r['site']==site for r in local)>=5]
        summaries.append(dict(subject=subject,matched_run=target,eligible_sites=len(eligible_sites),
            published_sites=len(np.atleast_1d(run['stimpair_names'])),detected_responses=len(values),
            pooled_median_ms=float(np.median(values)) if len(values) else None,
            pooled_iqr_ms=float(np.subtract(*np.quantile(values,[.75,.25]))) if len(values) else None,
            median_within_site_iqr_ms=float(np.median(site_iqrs)) if site_iqrs else None))
        if local:
            typical=min(local,key=lambda r:abs(r['latency_ms']-np.median(values)))
            examples.append(dict(typical,raw_header=item['path'].replace('.json','.vhdr'),fs=item['fs']))
    pd.DataFrame(edges).to_csv(out/'matched-latencies.csv',index=False)
    pd.DataFrame(sites).to_csv(out/'site-qc.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    (out/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(derivative_sha256=sha256(path),script_sha256=sha256(__file__),
        detector='Published V2 N1 samples, indexed through each run tt; no new detector fitted.',
        exclusions='Any pulse epoch -2..3 s intersects annotated artefact ±1 s or seizure ±300 s: exclude whole site; published averages cannot be cleaned retrospectively.',
        scope='Exactly matched runs and retained surface contacts; unequal stimulation coverage/intensity. Descriptive pilot, not an independent-edge association test.'),indent=2)+'\n')
    return summaries


def native_examples(root):
    import mne
    root=Path(root);out=root/'responses'
    selected=json.loads((out/'examples.json').read_text());arrays={}
    for row in selected:
        raw=mne.io.read_raw_brainvision(root/'raw'/row['raw_header'],preload=False,verbose='ERROR')
        events=pd.read_csv(root/'raw'/row['raw_header'].replace('_ieeg.vhdr','_events.tsv'),sep='\t')
        pulses=events[(events.trial_type=='electrical_stimulation') & events.electrical_stimulation_site.map(lambda s:site_key(s)==site_key(row['site']))]
        fs=row['fs'];epochs=[]
        # Match the published time-vector convention, including its one-based sample indexing.
        time=np.arange(1,5*fs+1)/fs-2
        for sample in pulses.sample_start:
            epoch=raw.get_data(picks=[row['response']],start=round(sample-2*fs),stop=round(sample+3*fs))[0]*1e6
            if len(epoch)==len(time):epochs.append(epoch-np.median(epoch[(time>-1)&(time<-.1)]))
        mean=np.mean(epochs,axis=0);mean-=np.median(mean[(time>-2)&(time<-.1)])
        visible=(time>=-.05)&(time<=.15)
        arrays[row['subject']+'_time']=time[visible]*1000
        arrays[row['subject']+'_mean']=mean[visible]
        arrays[row['subject']+'_epochs']=np.asarray(epochs)[:,visible]
    np.savez_compressed(out/'native-examples.npz',**arrays)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--native',action='store_true')
    parser.add_argument('--metadata-root',type=Path)
    args=parser.parse_args();print(json.dumps(match(args.root,args.metadata_root),indent=2))
    if args.native:native_examples(args.root)
