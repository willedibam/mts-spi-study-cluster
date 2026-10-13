"""Native-waveform QC and joint sampling/window audit for the bounded CCEP pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, iirnotch, resample_poly, sosfiltfilt, tf2sos, welch
from scipy.stats import rankdata

from research.empirical.ccep_pilot.acquire import sha256

OPTIONS = ((128, 45), (256, 100), (512, 200))
LAGS = (1 / 128, 4 / 128, 16 / 128)


def clean_windows(events, stop, duration, maximum=3):
    """Disjoint windows and four-second filter margins outside marked artefacts."""
    intervals = [(5., stop - 5.)]
    for row in events.sort_values('onset').itertuples():
        if row.trial_type not in ('artefact', 'artifact', 'seizure'):
            continue
        margin = 300. if row.trial_type == 'seizure' else 1.
        a, b = row.onset - margin, row.onset + max(row.duration, 0) + margin
        next_intervals = []
        for lo, hi in intervals:
            if b <= lo or a >= hi:
                next_intervals.append((lo, hi))
            else:
                if a > lo: next_intervals.append((lo, a))
                if b < hi: next_intervals.append((b, hi))
        intervals = next_intervals
    candidates = []
    for lo, hi in intervals:
        lo, hi = lo + 4., hi - 4.
        count = int((hi - lo) // duration)
        if count > 0:
            candidates.extend(np.linspace(lo, hi - duration, count).tolist())
    if len(candidates) <= maximum:
        return candidates
    return [candidates[i] for i in np.linspace(0, len(candidates)-1, maximum).round().astype(int)]


def lag_matrix(x, seconds, fs):
    tau = round(seconds * fs)
    if tau == 0: return np.corrcoef(x)
    m = len(x)
    cross = np.corrcoef(x[:, tau:], x[:, :-tau])[:m, m:]
    return (cross + cross.T) / 2


def profiles(x, fs):
    i, j = np.triu_indices(len(x), 1)
    result = {'pearson': np.corrcoef(x)[i, j],
              'spearman': np.corrcoef(rankdata(x, axis=1))[i, j]}
    for lag in LAGS:
        result[f'pearson_{lag*1000:g}ms'] = lag_matrix(x, lag, fs)[i, j]
    for low, high in ((4, 8), (8, 13), (13, 30), (30, 45)):
        y = sosfiltfilt(butter(4, (low, high), fs=fs, btype='bandpass', output='sos'), x, axis=1)
        analytic = hilbert(y, axis=1)[:, round(fs):-round(fs)]
        envelope = abs(analytic)
        unit = analytic / np.maximum(envelope, 1e-15)
        result[f'plv_{low}_{high}'] = abs(unit @ unit.conj().T / unit.shape[1])[i, j]
        result[f'envelope_{low}_{high}'] = np.corrcoef(envelope)[i, j]
    return result


def feature_vector(profile):
    table = np.stack(list(profile.values()))
    valid = np.isfinite(table).all(axis=1) & (table.std(axis=1) > 1e-8)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.corrcoef(table)
    z[~valid, :] = np.nan; z[:, ~valid] = np.nan
    return z[np.triu_indices(len(table), 1)]


def compare_profiles(a, b):
    names = list(a)
    rmse = [float(np.sqrt(np.mean((a[n]-b[n])**2))) for n in names]
    zdelta = abs(feature_vector(a) - feature_vector(b))
    return dict(edge_rmse_max=max(rmse), edge_rmse_median=float(np.median(rmse)),
                z_error_median=float(np.nanmedian(zdelta)), z_error_p95=float(np.nanquantile(zdelta, .95)),
                z_valid=int(np.isfinite(zdelta).sum()),
                **{f'edge_rmse_{n}': e for n, e in zip(names, rmse)})


def filtered(x, fs, cutoff):
    y = x.copy()
    for frequency in range(50, cutoff + 1, 50):
        b, a = iirnotch(frequency, 30, fs=fs)
        y = sosfiltfilt(tf2sos(b, a), y, axis=1)
    return sosfiltfilt(butter(4, (.5, cutoff), fs=fs, btype='bandpass', output='sos'), y, axis=1)


def scope(root, selection):
    root = Path(root)
    output = root / 'scope'; output.mkdir(parents=True, exist_ok=True)
    prepared = root / 'prepared'; prepared.mkdir(exist_ok=True)
    reports, audits, durations, quality, spectra, reference_rows, heatmaps = [], [], [], [], [], [], {}
    for item in selection['records']:
        subject = item['subject']
        prefix = item['path'].replace('_ieeg.json', '')
        channels = pd.read_csv(root/'raw'/(prefix+'_channels.tsv'), sep='\t')
        events = pd.read_csv(root/'raw'/(prefix+'_events.tsv'), sep='\t')
        metadata = json.loads((root/'raw'/item['path']).read_text())
        raw = mne.io.read_raw_brainvision(root/'raw'/(prefix+'_ieeg.vhdr'), preload=False, verbose='ERROR')
        fs = int(metadata['SamplingFrequency'])
        assert abs(raw.info['sfreq'] - fs) < .01
        assert raw.n_times == round(item['duration'] * fs)
        assert channels.name.tolist() == raw.ch_names
        pulse = events.query("trial_type == 'electrical_stimulation'")
        timing_error = float(np.max(abs(pulse.onset * fs - pulse.sample_start)))
        assert timing_error <= 1.01, timing_error
        stimulation = events[events.trial_type.isin(['stimulation', 'electrical_stimulation'])]
        stop = float(stimulation.onset.min())
        retained = channels.query("type == 'ECOG' and status == 'good'").name.tolist()
        picks = [raw.ch_names.index(name) for name in retained]
        candidate_starts = {d: clean_windows(events, stop, d) for d in (16, 32, 64)}
        starts = candidate_starts[32]
        if len(starts) < 2:
            reports.append(dict(subject=subject, status='insufficient clean 32-s intervals', available_windows=candidate_starts))
            continue
        arrays, windows, remove = {}, [], set()
        for index, start in enumerate(starts):
            native = raw.get_data(picks=picks, start=round((start-4)*fs), stop=round((start+36)*fs))
            rms = np.std(native, axis=1)
            for k in np.flatnonzero(~np.isfinite(native).all(axis=1) | (rms < 1e-10)):
                remove.add(retained[k])
            windows.append((index, start, native))
        keep = np.array([n not in remove for n in retained])
        retained = [n for n in retained if n not in remove]
        for index, start, native in windows:
            native = native[keep]
            central = native[:, 4*fs:-4*fs]
            centre = np.median(central, axis=1, keepdims=True)
            mad = np.median(abs(central-centre), axis=1, keepdims=True) * 1.4826
            robust = abs(central-centre) / np.maximum(mad, 1e-12)
            global_fraction = float(np.mean((robust > 12).mean(axis=0) >= .2))
            max_uv = float(abs(central).max()*1e6)
            qc_pass = global_fraction < .005 and max_uv < 5000 and np.isfinite(central).all()
            quality.append(dict(subject=subject, window=index, start=start, M=len(native),
                                rms_uv_median=float(np.median(central.std(axis=1))*1e6), max_uv=max_uv,
                                global_extreme_fraction=global_fraction, qc_pass=bool(qc_pass)))
            car = native - native.mean(axis=0, keepdims=True)
            f, psd = welch(car[:, 4*fs:-4*fs], fs=fs, nperseg=4*fs, axis=1)
            total = np.trapezoid(psd[:, (f>=.5)&(f<=200)], f[(f>=.5)&(f<=200)], axis=1)
            for cutoff in (45, 100, 200):
                mask = (f>cutoff)&(f<=200)
                tail = np.trapezoid(psd[:, mask], f[mask], axis=1) if mask.any() else np.zeros(len(car))
                spectra.append(dict(subject=subject, window=index, cutoff=cutoff,
                                    tail_fraction_median=float(np.median(tail/total)),
                                    tail_fraction_p95=float(np.quantile(tail/total,.95))))
            arrays[f'w{index}_psd_f'] = f
            arrays[f'w{index}_psd_median'] = np.median(psd, axis=0)
            ref = filtered(car, fs, 200)[:, 4*fs:-4*fs]
            ref_profiles = profiles(ref, fs)
            for rate, cutoff in OPTIONS:
                high = filtered(car, fs, cutoff)
                low = resample_poly(high, rate, fs, axis=1)[:, 4*rate:-4*rate]
                assert low.shape[1] == 32*rate
                same_band = profiles(high[:, 4*fs:-4*fs], fs)
                current = profiles(low, rate)
                arrays[f'w{index}_fs{rate}_bw{cutoff}'] = low
                audits.append(dict(subject=subject, window=index, start=start, M=len(low), fs=rate,
                                   bandwidth=cutoff, T=low.shape[1], comparison='resampling_same_band',
                                   **compare_profiles(current, same_band)))
                audits.append(dict(subject=subject, window=index, start=start, M=len(low), fs=rate,
                                   bandwidth=cutoff, T=low.shape[1], comparison='bandwidth_vs_200Hz',
                                   **compare_profiles(current, ref_profiles)))
                for half in (0, 1):
                    half_profiles = profiles(low[:, half*16*rate:(half+1)*16*rate], rate)
                    durations.append(dict(subject=subject, window=index, fs=rate, bandwidth=cutoff,
                                          duration=16, half=half, **compare_profiles(half_profiles, current)))
                if index == 0:
                    acquisition = resample_poly(filtered(native, fs, cutoff), rate, fs, axis=1)[:, 4*rate:-4*rate]
                    reference_rows.append(dict(subject=subject, fs=rate, bandwidth=cutoff,
                                               **compare_profiles(current, profiles(acquisition, rate))))
                    if rate == 256:
                        heatmaps[subject] = low[:, :4*rate]
            print(f'{subject} window {index}: M={len(retained)}, start={start:.3f}s, QC={qc_pass}', flush=True)
        arrays['channel_names'] = np.array(retained)
        arrays['starts'] = np.array(starts)
        np.savez_compressed(prepared/f'{subject}.npz', **arrays)
        reports.append(dict(subject=subject, status='audited', M=len(retained), native_fs=fs,
                            native_duration=item['duration'], first_pulse=float(pulse.onset.min()),
                            baseline_stop=stop, original_reference=metadata['iEEGReference'],
                            excluded_declared=channels.loc[(channels.type!='ECOG')|(channels.status!='good'),'name'].tolist(),
                            excluded_numeric=sorted(remove), channels=retained, starts=starts,
                            available_windows=candidate_starts, timing_error_samples=timing_error,
                            seizure_annotations=events.loc[events.trial_type=='seizure',['onset','duration']].to_dict('records')))
        # Save incrementally so an error does not discard completed participants.
        for name, rows in [('sampling',audits),('duration',durations),('quality',quality),('spectra',spectra),('reference',reference_rows)]:
            pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
        (output/'participants.json').write_text(json.dumps(reports,indent=2)+'\n')
        np.savez_compressed(output/'heatmaps.npz', **heatmaps)
    (output/'provenance.json').write_text(json.dumps(dict(selection_sha256=sha256(Path(__file__).parent/'configs/selection.json'),
        script_sha256=sha256(__file__), acquisition_sha256=sha256(root/'acquisition.json'),
        profile_names=list(ref_profiles), lags_seconds=LAGS, sampling_options=OPTIONS,
        note='Selected diagnostic families, not a full-p90 invariance claim. CAR over retained ECoG; no concatenation.'),indent=2)+'\n')


def long_window_audit(root, rate, cutoff):
    root = Path(root)
    selected = json.loads((Path(__file__).parent/'configs/selection.json').read_text())['records']
    report = json.loads((root/'scope/participants.json').read_text())
    records = []
    for row in report:
        starts = row.get('available_windows', {}).get('64', [])
        if not starts: continue
        item = next(r for r in selected if r['subject'] == row['subject'])
        raw = mne.io.read_raw_brainvision(root/'raw'/item['path'].replace('.json','.vhdr'), preload=False, verbose='ERROR')
        start, fs = starts[0], item['fs']
        x = raw.get_data(picks=row['channels'], start=round((start-4)*fs), stop=round((start+68)*fs))
        x -= x.mean(axis=0,keepdims=True)
        x = resample_poly(filtered(x,fs,cutoff),rate,fs,axis=1)[:,4*rate:-4*rate]
        whole = profiles(x,rate)
        for duration in (16,32):
            for index in range(64//duration):
                part = profiles(x[:,index*duration*rate:(index+1)*duration*rate],rate)
                records.append(dict(subject=row['subject'],start=start,fs=rate,bandwidth=cutoff,
                                    duration=duration,reference_seconds=64,part=index,**compare_profiles(part,whole)))
    pd.DataFrame(records).to_csv(root/'scope/long-duration.csv',index=False)


def build(root, rate, cutoff, seconds, reference_control=False):
    """Build homogeneous participant archives only after the audit is reviewed."""
    import yaml
    root=Path(root)
    reports=json.loads((root/'scope/participants.json').read_text())
    quality=pd.read_csv(root/'scope/quality.csv')
    assert (rate,cutoff) in OPTIONS and seconds in (16,32)
    for row in reports:
        if row['status']!='audited': continue
        subject=row['subject']; arrays={}; records=[]
        with np.load(root/'prepared'/f'{subject}.npz') as bank:
            for index,start in enumerate(row['starts']):
                if not bool(quality.query('subject == @subject and window == @index').qc_pass.iloc[0]): continue
                x=bank[f'w{index}_fs{rate}_bw{cutoff}'][:,:seconds*rate]
                x=(x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True)
                name=f'{subject}-w{index}-fs{rate}-t{x.shape[1]}'
                arrays[name]=x
                records.append(dict(name=name,subject=subject,window=index,start=start,M=len(x),T=x.shape[1],fs=rate,
                                    seconds=seconds,bandwidth=[.5,cutoff],reference='CAR retained good ECoG',channels=row['channels']))
        if len(records)<2: continue
        for record in records: record['reference_control'] = False
        if reference_control:
            selected=json.loads((Path(__file__).parent/'configs/selection.json').read_text())['records']
            item=next(r for r in selected if r['subject']==subject)
            raw=mne.io.read_raw_brainvision(root/'raw'/item['path'].replace('.json','.vhdr'),preload=False,verbose='ERROR')
            start=records[0]['start']; fs=item['fs']
            x=raw.get_data(picks=row['channels'],start=round((start-4)*fs),stop=round((start+seconds+4)*fs))
            x=resample_poly(filtered(x,fs,cutoff),rate,fs,axis=1)[:,4*rate:-4*rate]
            x=(x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True)
            name=records[0]['name']+'-acquisition-reference'
            arrays[name]=x
            records.append(dict(records[0],name=name,reference='Acquisition G2; probably mastoid',reference_control=True))
        folder=root/'corpora'/f'{subject}-fs{rate}-t{seconds*rate}';folder.mkdir(parents=True,exist_ok=True)
        arrays.update(__dataset_names__=np.array([r['name'] for r in records]),
                      __labels_json__=np.array([json.dumps([subject,f"window-{r['window']}"]) for r in records]),
                      __shapes__=np.array([(r['M'],r['T']) for r in records]),__axis_order__=np.array(['process','observation']))
        archive=folder/'observations.npz';np.savez_compressed(archive,**arrays)
        config=dict(name=f'ccep-{subject}-fs{rate}-t{seconds*rate}',
                    source=dict(format='named-npz-v1',archive=str(archive.resolve()),sha256=sha256(archive),axis_order=['process','observation']),
                    base_output_dir=str((folder/'mpis').resolve()),pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261014)
        (folder/'external.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        (folder/'manifest.json').write_text(json.dumps(dict(rows=records,archive_sha256=sha256(archive),
                scope_sha256=sha256(root/'scope/participants.json'),script_sha256=sha256(__file__)),indent=2)+'\n')
        print(folder/'external.yaml',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['scope','build','long-windows'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--rate',type=int);parser.add_argument('--cutoff',type=int);parser.add_argument('--seconds',type=int,default=32)
    parser.add_argument('--reference-control',action='store_true')
    args=parser.parse_args()
    if args.command=='scope': scope(args.root,json.loads((Path(__file__).parent/'configs/selection.json').read_text()))
    elif args.command=='long-windows': long_window_audit(args.root,args.rate,args.cutoff)
    else: build(args.root,args.rate,args.cutoff,args.seconds,args.reference_control)


if __name__=='__main__': main()
