"""Frozen-pilot preprocessing for run-keyed cohort windows; manual QC gates release."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch

from .acquire import sha256
from .prepare import clean_windows, filtered

CONFIG = Path(__file__).parent / 'configs/cohort-selection.json'


def native_qc(central):
    centre = np.median(central, axis=1, keepdims=True)
    mad = np.median(abs(central-centre), axis=1, keepdims=True)*1.4826
    robust = abs(central-centre)/np.maximum(mad, 1e-12)
    fraction = float(np.mean((robust > 12).mean(axis=0) >= .2))
    maximum = float(abs(central).max()*1e6)
    return dict(global_extreme_fraction=fraction, max_uv=maximum,
                qc_pass=bool(fraction < .005 and maximum < 5000 and np.isfinite(central).all()))


def prepare(root, raw_root):
    root, raw_root = Path(root), Path(raw_root)
    selection = json.loads(CONFIG.read_text())
    receipt = json.loads((raw_root/'acquisition.json').read_text())
    assert receipt['selection_sha256'] == sha256(CONFIG)
    verified = {r['path']: r['sha256'] for r in receipt['files']}
    assert verified == {r['path']: r['sha256'] for r in selection['files']}
    for sub in ('prepared', 'native', 'scope', 'figures', 'logs'):
        (root/sub).mkdir(parents=True, exist_ok=True)
    reports = []
    for item in selection['records']:
        run = item['run_id']; prefix = item['path'].replace('_ieeg.json', '')
        destination = root/'scope'/f'{run}.json'
        if destination.exists():
            report = json.loads(destination.read_text())
            assert report['selection_sha256'] == sha256(CONFIG)
            assert report['script_sha256'] == sha256(__file__)
            assert sha256(root/'prepared'/f'{run}.npz') == report['prepared_sha256']
            reports.append(report); continue
        channels = pd.read_csv(raw_root/'raw'/(prefix+'_channels.tsv'), sep='\t')
        events = pd.read_csv(raw_root/'raw'/(prefix+'_events.tsv'), sep='\t')
        meta = json.loads((raw_root/'raw'/item['path']).read_text())
        raw = mne.io.read_raw_brainvision(raw_root/'raw'/(prefix+'_ieeg.vhdr'), preload=False, verbose='ERROR')
        fs = int(meta['SamplingFrequency'])
        assert raw.info['sfreq'] == fs and raw.n_times == round(item['duration']*fs)
        assert raw.ch_names == channels.name.tolist()
        pulses = events.query("trial_type == 'electrical_stimulation'")
        timing = float(np.max(abs(pulses.onset*fs-pulses.sample_start)))
        assert timing <= 1.01
        stop = float(events[events.trial_type.isin(['stimulation', 'electrical_stimulation'])].onset.min())
        starts = clean_windows(events, stop, 32)
        assert starts == item['starts'] and starts
        names = channels.query("type == 'ECOG' and status == 'good'").name.tolist()
        windows = [raw.get_data(picks=names, start=round((s-4)*fs), stop=round((s+36)*fs)) for s in starts]
        keep = np.ones(len(names), dtype=bool)
        for x in windows:
            assert x.shape[1] == 40*fs
            keep &= np.isfinite(x).all(axis=1) & (x.std(axis=1) >= 1e-10)
        removed = [n for n,k in zip(names,keep) if not k]
        names = [n for n,k in zip(names,keep) if k]
        if len(names) < 3: raise ValueError(f'{run}: insufficient retained contacts')
        prepared, native, rows = {}, {}, []
        for w, (start, x) in enumerate(zip(starts, windows)):
            x = x[keep]; central = x[:,4*fs:-4*fs]; car = x-x.mean(axis=0,keepdims=True)
            low = resample_poly(filtered(car,fs,45),128,fs,axis=1)[:,512:-512]
            acq = resample_poly(filtered(x,fs,45),128,fs,axis=1)[:,512:-512]
            assert low.shape == (len(names),4096) and np.isfinite(low).all()
            prepared[f'w{w}'] = low; prepared[f'a{w}'] = acq; native[f'w{w}'] = x
            centre = np.median(central,axis=1,keepdims=True)
            mad = np.maximum(1.4826*np.median(abs(central-centre),axis=1,keepdims=True),1e-12)
            robust_peak = (abs(central-centre)/mad).max(axis=1)
            order = np.argsort(robust_peak)[-3:][::-1]
            prepared[f'trace{w}'] = central[order]; prepared[f'trace_names{w}'] = np.array(names)[order]
            f, psd = welch(car[:,4*fs:-4*fs],fs=fs,nperseg=4*fs,axis=1)
            band = (f >= .5) & (f <= 200); high = (f > 45) & (f <= 200)
            ratio = np.trapezoid(psd[:,high],f[high],axis=1)/np.trapezoid(psd[:,band],f[band],axis=1)
            prepared[f'freq{w}'] = f; prepared[f'psd{w}'] = np.median(psd,axis=0)
            rows.append(dict(window=w,start=start,**native_qc(central),
                             high_frequency_fraction_median=float(np.median(ratio)),
                             worst_native_robust_peak=float(robust_peak.max()),
                             worst_native_channel=names[order[0]]))
        prepared['channel_names'] = np.array(names)
        prepared['native_fs'] = np.array(fs)
        native['channel_names'] = np.array(names); native['native_fs'] = np.array(fs)
        native['starts'] = np.array(starts)
        np.savez_compressed(root/'prepared'/f'{run}.npz', **prepared)
        np.savez_compressed(root/'native'/f'{run}.npz', **native)
        report = dict(run_id=run,subject=item['subject'],path=item['path'],M=len(names),T=4096,
                      native_fs=fs,first_stimulation=stop,original_reference=meta.get('iEEGReference'),
                      channels=names,removed_numerical=removed,
                      excluded_declared=channels.loc[(channels.type!='ECOG')|(channels.status!='good'),'name'].tolist(),
                      windows=rows,timing_error_samples=timing,
                      selection_sha256=sha256(CONFIG),script_sha256=sha256(__file__),
                      pilot_preprocessing_sha256=sha256(Path(__file__).parent/'prepare.py'),
                      prepared_sha256=sha256(root/'prepared'/f'{run}.npz'),
                      native_sha256=sha256(root/'native'/f'{run}.npz'))
        destination.write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        print(run,len(names),[r['qc_pass'] for r in rows],flush=True)
    (root/'scope/runs.json').write_text(json.dumps(reports,indent=2)+'\n')
    gallery(root, reports)


def gallery(root, reports):
    from .plots import style
    import seaborn as sns
    plt = style(); windows = [(r,w) for r in reports for w in r['windows']]
    for page, offset in enumerate(range(0,len(windows),5)):
        items = windows[offset:offset+5]
        fig, axes = plt.subplots(len(items),2,figsize=(11,2.3*len(items)),squeeze=False)
        for (ax, trace), (r,w) in zip(axes,items):
            k=w['window']
            with np.load(root/'prepared'/f"{r['run_id']}.npz") as bank:
                x=bank[f'w{k}']; q25,q75=np.quantile(x,[.25,.75],axis=1)
                scaled=(x-np.median(x,axis=1,keepdims=True))/np.maximum(q75-q25,1e-12)[:,None]
                ax.imshow(scaled,origin='lower',aspect='auto',extent=(0,32,.5,len(x)+.5),vmin=-3,vmax=3,
                          cmap=sns.color_palette('icefire',as_cmap=True),interpolation='nearest')
                ax.set(title=r['run_id'].replace('sub-ccepAgeUMCU','').replace('_ses-1_task-SPESclin','')+f' · w{k}',xlabel='Time (s)',ylabel='Channel')
                for y,name in zip(bank[f'trace{k}'],bank[f'trace_names{k}']):
                    trace.plot(np.arange(len(y))/r['native_fs'],y*1e6,lw=.3,label=name)
                trace.legend();trace.set(xlabel='Time (s)',ylabel='Native µV',title=f"Numeric QC {w['qc_pass']} · peak {w['worst_native_robust_peak']:.1f} robust SD")
        fig.savefig(root/'figures'/f'qc-{page:02}.png',dpi=150)
        fig.savefig(root/'figures'/f'qc-{page:02}.pdf')
        plt.close(fig)


def build(root, review):
    import yaml
    root=Path(root);review_path=Path(review);decisions=json.loads(review_path.read_text())
    reports=json.loads((root/'scope/runs.json').read_text());configs=[]
    assert decisions['scope_sha256']==sha256(root/'scope/runs.json')
    for report in reports:
        run=report['run_id'];decision=decisions['runs'][run]
        assert sha256(root/'prepared'/f'{run}.npz')==report['prepared_sha256']
        assert set(decision)=={str(w['window']) for w in report['windows']}
        retained=[w for w in report['windows'] if decision[str(w['window'])]['accept']]
        assert all(w['qc_pass'] for w in retained), 'Visual review cannot override numerical QC'
        if not retained:continue
        folder=root/'corpora'/run;folder.mkdir(parents=True,exist_ok=True)
        config=folder/'external.yaml'
        if config.exists():
            manifest=json.loads((folder/'manifest.json').read_text())
            assert manifest['review_sha256']==sha256(review_path)
            assert manifest['prepared_sha256']==report['prepared_sha256']
            assert manifest['archive_sha256']==sha256(folder/'observations.npz')
            configs.append(str(config));continue
        arrays={};records=[]
        with np.load(root/'prepared'/f'{run}.npz') as bank:
            for w,control in [(w,False) for w in retained]+[(retained[0],True)]:
                index=w['window'];x=bank[f"{'a' if control else 'w'}{index}"]
                assert np.isfinite(x).all() and (x.std(axis=1)>0).all()
                x=(x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True)
                name=f"{run}-w{index}"+('-acquisition-reference' if control else '-car')
                arrays[name]=x
                records.append(dict(name=name,subject=report['subject'],run_id=run,window=index,start=w['start'],
                                    M=len(x),T=4096,fs=128,seconds=32,bandwidth=[.5,45],channels=report['channels'],
                                    reference_control=control,reference=report['original_reference'] if control else 'CAR retained good ECoG'))
        arrays.update(__dataset_names__=np.array([r['name'] for r in records]),
                      __labels_json__=np.array([json.dumps([r['subject'],run,f"window-{r['window']}"]) for r in records]),
                      __shapes__=np.array([(r['M'],r['T']) for r in records]),__axis_order__=np.array(['process','observation']))
        archive=folder/'observations.npz'
        if archive.exists():raise FileExistsError(archive)
        np.savez_compressed(archive,**arrays)
        payload=dict(name='ccep-'+run,source=dict(format='named-npz-v1',archive=str(archive.resolve()),sha256=sha256(archive),axis_order=['process','observation']),
                     base_output_dir=str((folder/'mpis').resolve()),pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261014)
        config.write_text(yaml.safe_dump(payload,sort_keys=False))
        (folder/'manifest.json').write_text(json.dumps(dict(rows=records,archive_sha256=sha256(archive),
                  prepared_sha256=report['prepared_sha256'],review_sha256=sha256(review_path)),indent=2)+'\n')
        configs.append(str(config))
    (root/'active-configs.json').write_text(json.dumps(configs,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','build'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--raw-root',type=Path);parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.root,args.raw_root)
    else: build(args.root,args.review)
