"""Freeze CHB-MIT models and prepare independently annotated Siena confirmation.

Ambiguous event clocks are not silently repaired. They exclude ictal selection;
their full plausible interval still excludes seizure-free control windows.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

from mtsspi.provenance import atomic_json, atomic_savez, file_sha256
from research.representation.dataset_screen.prepare import CHB_CHANNELS, checksums, download, epoch, univariate, control_windows
from research.representation.empirical_transfer.prepare import write_corpus

BASE='https://physionet.org/files/siena-scalp-eeg/1.0.0/'
NODES=sorted({n for pair in CHB_CHANNELS for n in pair.split('-')})
ALIASES={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
# Published annotation spelling differs from the checksum inventory. PN01 and
# PN11 each have one EDF; the registration clock is also checked against it.
FILE_ALIASES={'PN01.edf':'PN01-1.edf','PN11-.edf':'PN11-1.edf',
              **{f'PNO6-{i}.edf':f'PN06-{i}.edf' for i in [1,2,4]}}


def clocks(value):
    compact=re.sub(r'\s+','',value)
    found=re.findall(r'(?<!\d)(\d{1,2})[.:](\d{2})[.:](\d{2})(?!\d)',compact)
    result=[]
    for fields in found:
        h,m,s=map(int,fields)
        if h>=24 or m>=60 or s>=60: raise ValueError('invalid annotation clock')
        result.append(h*3600+m*60+s)
    if not result: raise ValueError('annotation clock absent')
    return sorted(set(result))


def annotations(text):
    files=OrderedDict()
    def fields(block):
        return {k.lower():v.strip() for k,v in re.findall(
            r'(?im)^\s*(File name|Registration start time|(?:Seizure )?start time|(?:Seizure )?end time)\s*:\s*([^\r\n]+)',block)}
    blocks=re.split(r'(?im)^\s*Seizure\s+n\s*[^\r\n]*[\r\n]+',text)
    header=fields(blocks[0]);previous=header.get('file name','')
    for block in blocks[1:]:
        values=fields(block)
        original=values.get('file name',previous);previous=original
        name=FILE_ALIASES.get(original,original)
        if not re.fullmatch(r'PN\d{2}-[\d.]+\.edf',name): raise ValueError(f'unsafe/unknown Siena filename: {name}')
        item=files.setdefault(name,dict(name=name,events=[],issues=[],annotation_filenames=[]))
        if original not in item['annotation_filenames']:item['annotation_filenames'].append(original)
        try:
            registration_text=values.get('registration start time',item.get('registration_text'))
            if registration_text is None and original==header.get('file name'):
                registration_text=header.get('registration start time')
            if registration_text is None:raise ValueError('registration start absent for this file')
            registration=clocks(registration_text)
            if len(registration)!=1: raise ValueError('ambiguous registration start')
            item['registration_text']=registration_text
            start=values.get('seizure start time',values.get('start time',''))
            end=values.get('seizure end time',values.get('end time',''))
            starts,ends=clocks(start),clocks(end)
            item['events'].append(dict(registration=registration[0],starts=starts,ends=ends,
                unambiguous=len(starts)==len(ends)==1,
                original_start=start,original_end=end))
        except (ValueError,KeyError) as error:
            item['issues'].append(str(error))
    if not files: raise ValueError('no Siena seizure annotations parsed')
    return list(files.values())


def relative_events(item,start_clock,duration):
    if item['issues'] or not item['events']: raise ValueError('unresolved annotation in file')
    all_events,eligible=[],[]
    for event in item['events']:
        if event['registration']!=start_clock: raise ValueError('EDF/annotation start clocks differ')
        starts=[(v-start_clock)%86400 for v in event['starts']]
        ends=[(v-start_clock)%86400 for v in event['ends']]
        a,b=min(starts),max(ends)
        if not 0<=a<b<=duration: raise ValueError('annotated event outside EDF or reversed')
        all_events.append((a,b))
        if event['unambiguous'] and b-a>=20: eligible.append((a,b))
    return all_events,eligible[:2]


def montage(names):
    mapping={}
    for i,name in enumerate(names):
        if not name.upper().startswith('EEG '): continue
        label=name.upper().removeprefix('EEG ').strip()
        label=ALIASES.get(label,label)
        if label in NODES:
            if label in mapping: raise ValueError(f'duplicate scalp electrode: {label}')
            mapping[label]=i
    if set(mapping)!=set(NODES): raise ValueError('required scalp electrode absent')
    return [mapping[name] for name in NODES]


def bipolar(x):
    lookup={name:i for i,name in enumerate(NODES)}
    return np.array([x[:,lookup[a]]-x[:,lookup[b]] for a,b in (pair.split('-') for pair in CHB_CHANNELS)]).T


def streamed_edf(name,hashes,root):
    path=root/name
    if path.exists():
        if file_sha256(path)!=hashes[name]: raise ValueError('cached Siena EDF checksum mismatch')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial');digest=hashlib.sha256()
    with urllib.request.urlopen(BASE+name,timeout=120) as response, partial.open('wb') as output:
        while chunk:=response.read(1024*1024): output.write(chunk);digest.update(chunk)
    if digest.hexdigest()!=hashes[name]: raise ValueError('published Siena EDF checksum mismatch')
    partial.replace(path)
    return path


def freeze(root,source):
    path=root/'source-lock.json'
    if path.exists():
        lock=json.loads(path.read_text())
        for name,digest in lock['files'].items():
            if file_sha256(source/name)!=digest: raise ValueError('frozen CHB-MIT artifact changed')
        return lock
    original=json.loads((source/'statistical-source-lock.json').read_text())
    names=['manifest.json','bank.json','statistical-source-lock.json']
    models={name:dict(kind='statistical',report=f'screen-models/{name}.json',weights=[f'screen-models/{name}.joblib'])
            for name in original['models']}
    for method in ['minirocket','minirocket_extended','inceptiontime','aligned_encoder']:
        directory='rocket-extended' if method=='minirocket_extended' else 'screen-models'
        base='minirocket' if method=='minirocket_extended' else method
        report_path=f'{directory}/{base}.json';report=json.loads((source/report_path).read_text())
        if report['identity']['target_data_used'] is not False: raise ValueError('source provenance missing')
        weights=[]
        for member in report['members']:
            relative=f'{directory}/{member["path"]}'
            if file_sha256(source/relative)!=member['sha256']: raise ValueError('source member changed')
            weights.append(relative)
        models[method]=dict(kind='minirocket' if method.startswith('minirocket') else 'neural',
                            report=report_path,weights=weights)
    for name,model in models.items():
        names+=[model['report'],*model['weights']]
        if model['kind']=='statistical' and file_sha256(source/model['weights'][0])!=original['models'][name]:
            raise ValueError('source statistical model changed')
    lock=dict(source_root=str(source),files={name:file_sha256(source/name) for name in names},models=models,
        protocol=dict(dataset=BASE,target='annotated ictal versus seizure-free, not seizure prediction',
            M=16,T=2000,rate_hz=100,filter_hz=[.5,40],seconds=20,filter_context_seconds=10,
            bipolar_channels=CHB_CHANNELS,maximum_files_per_person=2,maximum_events_per_file=2,
            control_margin_seconds=300,selection='first two eligible files in annotation order; central 20 seconds; ambiguous clocks excluded as positive events and conservatively excluded from controls',
            comparison='All existing source-frozen statistical/raw models, plus source-only MiniRocket ridge extension. No Siena fitting, threshold calibration, or target-driven model selection.',
            primary='m_plus_z versus m Brier; report z versus m and raw comparators alongside, with BA secondary',
            uncertainty='Paired whole-patient bootstrap conditional on fitted models; finite cohort and exploratory discovery history remain limitations.'),
        preparation_sha256=file_sha256(Path(__file__)))
    atomic_json(path,lock)
    return lock


def prepare(root,source):
    import mne
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): raise FileExistsError('Siena corpus already prepared')
    lock=freeze(root,source)
    raw_root=root/'raw';raw_root.mkdir(exist_ok=True)
    hashes=checksums(BASE,raw_root)
    summaries=sorted(name for name in hashes if re.fullmatch(r'PN\d{2}/Seizures-list-PN\d{2}\.txt',name))
    if len(summaries)!=14: raise ValueError('unexpected Siena patient inventory')
    for name in ['subject_info.csv','RECORDS','LICENSE.txt']:
        if name in hashes:download(BASE,name,hashes,raw_root)
    parsed={}
    for name in summaries:
        person=Path(name).parent.name
        parsed[name]=annotations(download(BASE,name,hashes,raw_root).read_text())
        for item in parsed[name]:
            if person+'/'+item['name'] not in hashes:raise ValueError('annotated EDF absent from published inventory')
    def person_task(name):
        person=Path(name).parent.name
        files=parsed[name];rows,arrays,issues=[],{},[];accepted=0
        for item in files:
            if accepted==2:break
            relative=person+'/'+item['name']
            if relative not in hashes:raise ValueError('annotated EDF absent from published inventory')
            if item['issues']:
                issues.append(dict(file=relative,reason='unresolved annotation fields',details=item['issues']));continue
            if not any(e['unambiguous'] and 20<=(e['ends'][0]-e['starts'][0])%86400 for e in item['events']):
                issues.append(dict(file=relative,reason='no unambiguous event of at least 20 seconds'));continue
            path=streamed_edf(relative,hashes,raw_root)
            raw=mne.io.read_raw_edf(path,preload=False,verbose='ERROR')
            try:
                with path.open('rb') as inp: header=inp.read(256)
                start=clocks(header[176:184].decode())[0]
                if raw.info['sfreq']!=512:raise ValueError('unexpected Siena sampling rate')
                picks=montage(raw.ch_names)
                events,selected=relative_events(item,start,raw.n_times/512)
                controls=control_windows(events,raw.n_times/512,len(selected),seconds=20,margin=300)
                if len(controls)!=len(selected) or not selected:raise ValueError('no eligible matched ictal/control intervals')
                added=0
                for pair,((a,b),control) in enumerate(zip(selected,controls)):
                    pair_rows,pair_arrays=[],{}
                    try:
                        for label,onset in [(1,(a+b-20)/2),(0,control)]:
                            x=bipolar(epoch(raw,onset,20,picks,100,.5,40));univariate(x)
                            key=f'{person}-{Path(item["name"]).stem}-event{pair}-state{label}'
                            pair_arrays[key]=x
                            pair_rows.append(dict(row_id=key,group=person,role='confirmation',target=label,
                                native_file=relative,native_sha256=hashes[relative],onset_seconds=onset,
                                source_electrodes=[raw.ch_names[i] for i in picks],montage_rank=int(np.linalg.matrix_rank(x))))
                    except ValueError as error:
                        issues.append(dict(file=relative,pair=pair,reason=str(error)));continue
                    rows.extend(pair_rows);arrays.update(pair_arrays);added+=1
                if added:accepted+=1
            except ValueError as error:issues.append(dict(file=relative,reason=str(error)))
            finally:raw.close()
        if not rows:issues.append(dict(group=person,reason='no eligible paired observations'))
        print('Prepared Siena',person,len(rows),flush=True)
        return rows,arrays,issues
    rows,physical,issues=[],{},[]
    with ThreadPoolExecutor(2) as pool:
        for r,x,i in pool.map(person_task,summaries):rows.extend(r);physical.update(x);issues.extend(i)
    if not rows or len(rows)>112:raise ValueError('empty/oversized external corpus')
    normalized={key:(x-x.mean(0))/x.std(0) for key,x in physical.items()}
    provenance=dict(dataset=BASE,source_lock_sha256=file_sha256(root/'source-lock.json'),
        preparation_sha256=file_sha256(Path(__file__)),protocol=lock['protocol'],
        normalization='Same as CHB-MIT: within-record standardization for SPIs, volts retained for waveform models and univariate features.',
        montage='Named EEG monopoles differenced into the fixed CHB-MIT bipolar order; T3/T4/T5/T6 aliases T7/T8/P7/P8. No guessing O1 from the annotation typo: actual EDF EEG O1 label is required.',
        annotation_filename_aliases=FILE_ALIASES,
        limitation='Curated events at least 20 seconds, matched seizure-free controls; no continuous-stream/clinical-prevalence claim. Bipolar montage has linear constraints; rank recorded per observation.')
    write_corpus(root,'siena-confirmation-r10-261102',rows,normalized,provenance)
    atomic_savez(root/'raw-views.npz',{key:x.astype(np.float32) for key,x in physical.items()})
    atomic_savez(root/'univariate.npz',dict(u=np.array([univariate(physical[r['row_id']]) for r in rows]),row_id=np.array([r['row_id'] for r in rows])))
    atomic_json(root/'input-audit.json',dict(provenance=provenance,issues=issues,records=len(rows),
        groups=sorted({r['group'] for r in rows}),raw_views_sha256=file_sha256(root/'raw-views.npz'),
        univariate_sha256=file_sha256(root/'univariate.npz')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();prepare(args.root,args.source)
