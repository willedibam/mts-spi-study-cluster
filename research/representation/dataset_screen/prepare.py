"""Download directly on Gadi and prepare a bounded discovery corpus per dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
from scipy import signal, stats
from scipy.io import loadmat
import yaml

from mtsspi.provenance import atomic_json, atomic_savez, file_sha256
from research.representation.empirical_transfer.prepare import write_corpus

PROTOCOL = Path('research/representation/dataset_screen/protocol.yaml')
SEED = 261031
CHB_CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2']
HMC_CHANNELS = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2',
                'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG']


def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'mts-spi-research'}), timeout=90) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def checksums(base, directory):
    path = directory / 'SHA256SUMS.txt'
    if not path.exists():
        path.write_bytes(fetch(base + 'SHA256SUMS.txt'))
    result = {}
    for line in path.read_text().splitlines():
        if re.match(r'^[0-9a-f]{64}\s', line):
            digest, name = line.split(maxsplit=1)
            result[name.lstrip('*').removeprefix('./')] = digest
    if not result:
        raise ValueError('no published checksums')
    return result


def verified_bytes(base, name, hashes):
    data = fetch(base + name)
    if hashlib.sha256(data).hexdigest() != hashes[name]:
        raise ValueError(f'published checksum mismatch: {name}')
    return data


def download(base, name, hashes, root):
    path = root / name
    if '..' in Path(name).parts or Path(name).is_absolute():
        raise ValueError('unsafe source member')
    if path.exists():
        if file_sha256(path) != hashes[name]:
            raise ValueError(f'cached input changed: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = verified_bytes(base, name, hashes)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_bytes(data)
    partial.replace(path)
    return path


def split_groups(groups, source_count, screen_count, seed=SEED):
    ordered = np.random.default_rng(seed).permutation(sorted(set(groups))).tolist()
    return dict(source=ordered[:source_count], screen=ordered[source_count:source_count + screen_count],
                confirmation_reserved=ordered[source_count + screen_count:], seed=seed)


def univariate(x):
    centered = x - x.mean(0)
    sd = x.std(0)
    if not np.isfinite(x).all() or np.any(sd <= 1e-14):
        raise ValueError('nonfinite or flat channel')
    power = np.abs(np.fft.rfft(centered, axis=0)[1:]) ** 2 / len(x)
    bands = np.stack([v.mean(0) for v in np.array_split(power, 32)])
    features = [x.mean(0), np.log(sd), stats.skew(x, axis=0), stats.kurtosis(x, axis=0),
                *np.quantile(x, [.1, .25, .5, .75, .9], axis=0),
                np.mean(np.abs(np.diff(x, axis=0)), axis=0),
                np.mean(np.diff(np.signbit(centered), axis=0), axis=0)]
    return np.concatenate([np.array(features).ravel(), np.log(np.maximum(bands, np.finfo(float).tiny)).ravel(),
                           (bands / np.maximum(bands.sum(0), np.finfo(float).tiny)).ravel()])


def finalize(root, dataset, rows, physical, provenance, audit):
    if not rows or len(rows) > 1500:
        raise ValueError('empty or oversized discovery corpus')
    normalized, raw, u = {}, {}, []
    for row in rows:
        x = np.asarray(physical[row['row_id']], dtype=np.float64)
        u.append(univariate(x))
        raw[row['row_id']] = x.astype(np.float32)
        normalized[row['row_id']] = (x - x.mean(0)) / x.std(0)
    provenance.update(protocol_sha256=file_sha256(PROTOCOL), preparation_sha256=file_sha256(Path(__file__)),
                      scope='Discovery only. No reserved confirmation waveform is included.',
                      normalization='Per-record channel z-score for SPI computation. Physical-unit waveforms and amplitude/spectral summaries retained for competitors.')
    write_corpus(root, f'{dataset}-discovery-r10-261031', rows, normalized, provenance)
    atomic_savez(root / 'raw-views.npz', raw)
    atomic_savez(root / 'univariate.npz', dict(u=np.array(u), row_id=np.array([r['row_id'] for r in rows])))
    atomic_json(root / 'input-audit.json', dict(provenance=provenance, issues=audit,
        raw_views_sha256=file_sha256(root / 'raw-views.npz'), univariate_sha256=file_sha256(root / 'univariate.npz'),
        counts={role: sum(r['role'] == role for r in rows) for role in ['source', 'screen']},
        groups={role: sorted({r['group'] for r in rows if r['role'] == role}) for role in ['source', 'screen']}))


def epoch(raw, onset, seconds, picks, rate, low, high, emg_index=None):
    native = float(raw.info['sfreq'])
    start = max(0, int(round((onset - 10) * native)))
    stop = min(raw.n_times, int(round((onset + seconds + 10) * native)))
    x = raw.get_data(picks=picks, start=start, stop=stop).T
    filtered = np.empty_like(x)
    for channel in range(x.shape[1]):
        hp = 10 if channel == emg_index else low
        sos = signal.butter(4, [hp, high], fs=native, btype='bandpass', output='sos')
        filtered[:, channel] = signal.sosfiltfilt(sos, x[:, channel])
    ratio = Fraction(rate / native).limit_denominator(1000)
    filtered = signal.resample_poly(filtered, ratio.numerator, ratio.denominator, axis=0)
    first = int(round((onset - start / native) * rate))
    result = filtered[first:first + int(seconds * rate)]
    if result.shape != (int(seconds * rate), len(picks)):
        raise ValueError('epoch extends beyond contiguous recording')
    univariate(result)  # label-independent finite/flat-channel gate
    return result


def parse_sleep(text):
    rows = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    result = {0: [], 1: []}
    for row in rows:
        label = row['Annotation'].strip()
        if label in ['Sleep stage N2', 'Sleep stage R'] and float(row['Duration']) == 30:
            result[int(label == 'Sleep stage R')].append(float(row['Recording onset']))
    return result


def hmc(root, spec):
    import mne
    base = 'https://physionet.org/files/hmc-sleep-staging/1.1/'
    raw_root = root / 'raw'; raw_root.mkdir(parents=True, exist_ok=True)
    hashes = checksums(base, raw_root)
    ids = sorted({Path(name).stem for name in hashes if re.fullmatch(r'recordings/SN\d{3}\.edf', name)})
    if len(ids) != 151 or set(ids) & {'SN014', 'SN064', 'SN135'}:
        raise ValueError('unexpected HMC release inventory')
    split = split_groups(ids, spec['source_groups'], spec['screen_groups'])
    atomic_json(root / 'split.json', split)
    tasks = [(person, role) for role in ['source', 'screen'] for person in split[role]]
    def person_task(task):
        person, role = task
        annotation = download(base, f'recordings/{person}_sleepscoring.txt', hashes, raw_root)
        labels = parse_sleep(annotation.read_text())
        if any(len(v) < spec['epochs_per_state'] for v in labels.values()):
            return [], {}, [dict(group=person, reason='insufficient annotated N2 or REM epochs')]
        path = download(base, f'recordings/{person}.edf', hashes, raw_root)
        raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
        if set(raw.ch_names) != set(HMC_CHANNELS) or raw.info['sfreq'] != 256:
            raw.close()
            return [], {}, [dict(group=person, reason='required HMC channel names/rate unavailable')]
        picks = [raw.ch_names.index(name) for name in HMC_CHANNELS]
        rows, arrays, issues = [], {}, []
        for label, onsets in labels.items():
            chosen = np.unique(np.round(np.linspace(0, len(onsets) - 1, spec['epochs_per_state'])).astype(int))
            for index in chosen:
                onset = onsets[index]
                key = f'{person}-state{label}-epoch{int(onset/30):04d}'
                try:
                    x = epoch(raw, onset, 30, picks, 128, .3, 55, emg_index=4)
                except ValueError as error:
                    issues.append(dict(row_id=key, reason=str(error))); continue
                arrays[key] = x
                rows.append(dict(row_id=key, group=person, role=role, target=label, onset_seconds=onset,
                    native_file=f'recordings/{person}.edf', native_sha256=hashes[f'recordings/{person}.edf']))
        raw.close()
        if set(r['target'] for r in rows) != {0, 1}:
            return [], {}, issues + [dict(group=person, reason='one state lost to waveform QC')]
        print('Prepared HMC', person, len(rows), flush=True)
        return rows, arrays, issues
    rows, arrays, issues = [], {}, []
    with ThreadPoolExecutor(4) as pool:
        for r, a, i in pool.map(person_task, tasks): rows.extend(r); arrays.update(a); issues.extend(i)
    finalize(root, 'hmc', rows, arrays, dict(dataset=base, split=split, channels=HMC_CHANNELS,
        sample_rate_hz=128, independent_unit='night ID; repeated-patient identity is not independently available'), issues)


def bipolar_emg(values, names):
    lookup = {name: i for i, name in enumerate(names)}
    forearm = [values[:, lookup[f'F{i}']] - values[:, lookup[f'F{i+8}']] for i in range(1, 9)]
    wrist = [values[:, lookup[f'W{i}']] - values[:, lookup[f'W{i+6}']] for i in range(1, 7)]
    return np.array(forearm + wrist).T


def grabmyo(root, spec):
    import wfdb
    base = 'https://physionet.org/files/grabmyo/1.1.0/'
    raw_root = root / 'raw'; raw_root.mkdir(parents=True, exist_ok=True)
    hashes = checksums(base, raw_root)
    split = split_groups([f'p{i:02d}' for i in range(1, 44)], 24, 8)
    atomic_json(root / 'split.json', split)
    tasks = [(person, role, 1 if role == 'source' else 2) for role in ['source', 'screen'] for person in split[role]]
    def person_task(task):
        person, role, session = task
        participant = int(person[1:])
        archive = raw_root / f'{person}-session{session}.zip'
        records = [f'Session{session}/session{session}_participant{participant}/session{session}_participant{participant}_gesture{g}_trial1' for g in range(1, 18)]
        if not archive.exists():
            partial = archive.with_suffix('.partial')
            with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as bundle:
                for record in records:
                    for extension in ['.hea', '.dat']:
                        name = record + extension
                        bundle.writestr(name, verified_bytes(base, name, hashes))
            partial.replace(archive)
        rows, arrays = [], {}
        with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(dir=os.environ.get('PBS_JOBFS')) as temporary:
            for gesture, record in enumerate(records, 1):
                local = Path(temporary) / Path(record).name
                for extension in ['.hea', '.dat']:
                    data = bundle.read(record + extension)
                    if hashlib.sha256(data).hexdigest() != hashes[record + extension]:
                        raise ValueError('GRABMyo cache checksum mismatch')
                    local.with_suffix(extension).write_bytes(data)
                data = wfdb.rdrecord(str(local), sampfrom=4096, sampto=6144)
                if data.fs != 2048 or data.p_signal.shape != (2048, 32):
                    raise ValueError('unexpected GRABMyo shape/rate')
                x = bipolar_emg(data.p_signal, data.sig_name)
                univariate(x)
                key = f'{person}-session{session}-gesture{gesture:02d}-trial1'
                rows.append(dict(row_id=key, group=person, role=role, target=gesture - 1,
                    gesture=gesture, session=session, trial=1, native_file=record,
                    native_sha256=hashes[record + '.dat']))
                arrays[key] = x
                for extension in ['.hea', '.dat']: local.with_suffix(extension).unlink()
        print('Prepared GRABMyo', person, session, flush=True)
        return rows, arrays
    rows, arrays = [], {}
    with ThreadPoolExecutor(4) as pool:
        for r, a in pool.map(person_task, tasks): rows.extend(r); arrays.update(a)
    finalize(root, 'grabmyo', rows, arrays, dict(dataset=base, split=split, sample_rate_hz=2048,
        independent_unit='participant', shift='source day 1 to unseen participants on day 2',
        channels='14 anatomical bipolar pairs; named mapping excludes U1-U4', crop_seconds=[2, 3]), [])


def parse_seizures(text):
    result = []
    for block in re.split(r'File Name:\s*', text)[1:]:
        name = block.splitlines()[0].strip()
        # The release includes both chb02_16+.edf and chb17a_03.edf.
        # Require a safe basename; published checksums establish file identity.
        if not re.fullmatch(r'chb[A-Za-z0-9_+]+\.edf', name):
            raise ValueError(f'unexpected EEG filename: {name}')
        starts = [float(v) for v in re.findall(r'Seizure(?:\s+\d+)? Start Time:\s*(\d+) seconds', block)]
        ends = [float(v) for v in re.findall(r'Seizure(?:\s+\d+)? End Time:\s*(\d+) seconds', block)]
        count = int(re.search(r'Number of Seizures in File:\s*(\d+)', block).group(1))
        if not len(starts) == len(ends) == count or any(e <= s for s, e in zip(starts, ends)):
            raise ValueError(f'inconsistent seizure annotations: {name}')
        result.append(dict(name=name, events=list(zip(starts, ends))))
    return result


def control_windows(events, duration, count, seconds=20, margin=300):
    boundaries = sorted([(max(0, a - margin), min(duration, b + margin)) for a, b in events])
    intervals, cursor = [], 10.
    for start, stop in boundaries:
        if start - cursor >= seconds: intervals.append((cursor, start))
        cursor = max(cursor, stop)
    if duration - 10 - cursor >= seconds: intervals.append((cursor, duration - 10))
    candidates = [(a, b) for a, b in intervals if b - a >= seconds]
    candidates.sort(key=lambda v: (-(v[1] - v[0]), v[0]))
    onsets = []
    for a, b in candidates:
        for onset in np.arange(a + (b - a) / 2 - seconds / 2, b - seconds + 1e-6, seconds):
            if len(onsets) == count: return onsets
            onsets.append(float(onset))
    return onsets[:count]


def chbmit(root, spec):
    import mne
    base = 'https://physionet.org/files/chbmit/1.0.0/'
    raw_root = root / 'raw'; raw_root.mkdir(parents=True, exist_ok=True)
    hashes = checksums(base, raw_root)
    people = [f'chb{i:02d}' for i in range(1, 25) if i != 21]
    split = split_groups(people, 16, 7)
    split['external_confirmation_reserved'] = 'Siena Scalp EEG, not accessed by this preparation'
    atomic_json(root / 'split.json', split)
    def person_task(task):
        person, role = task
        name = f'{person}/{person}-summary.txt'
        if name not in hashes:
            return [], {}, [dict(group=person, reason='published summary absent; no inferred annotations')]
        summary = download(base, name, hashes, raw_root)
        files = [f for f in parse_seizures(summary.read_text()) if any(b - a >= 20 for a, b in f['events'])]
        rows, arrays, issues, used_files = [], {}, [], 0
        for item in files:
            if used_files == 2: break
            relative = person + '/' + item['name']
            path = download(base, relative, hashes, raw_root)
            raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
            canonical = [re.sub(r'-\d+$', '', n.upper()) for n in raw.ch_names]
            if any(name not in canonical for name in CHB_CHANNELS) or raw.info['sfreq'] != 256:
                issues.append(dict(file=relative, reason='required bipolar montage or rate unavailable')); raw.close(); continue
            picks = [canonical.index(name) for name in CHB_CHANNELS]
            ictal = [(a + b - 20) / 2 for a, b in item['events'] if b - a >= 20][:2]
            controls = control_windows(item['events'], raw.n_times / 256, len(ictal))
            if len(controls) < len(ictal):
                issues.append(dict(file=relative, reason='insufficient same-file seizure-free control interval')); raw.close(); continue
            new_rows, new_arrays = [], {}
            for pair, (seizure, control) in enumerate(zip(ictal, controls)):
                for label, onset in [(1, seizure), (0, control)]:
                    key = f'{Path(item["name"]).stem}-event{pair}-state{label}'
                    x = epoch(raw, onset, 20, picks, 100, .5, 40)
                    new_rows.append(dict(row_id=key, group=person, role=role, target=label,
                        native_file=relative, native_sha256=hashes[relative], onset_seconds=onset,
                        parent_event=f'{relative}-event{pair}'))
                    new_arrays[key] = x
            rows.extend(new_rows); arrays.update(new_arrays); used_files += 1; raw.close()
        print('Prepared CHB-MIT', person, len(rows), flush=True)
        return rows, arrays, issues
    tasks = [(person, role) for role in ['source', 'screen'] for person in split[role]]
    rows, arrays, issues = [], {}, []
    with ThreadPoolExecutor(4) as pool:
        for r, a, i in pool.map(person_task, tasks): rows.extend(r); arrays.update(a); issues.extend(i)
    finalize(root, 'chbmit', rows, arrays, dict(dataset=base, split=split, sample_rate_hz=100,
        independent_unit='person; duplicate case chb21 excluded', channels=CHB_CHANNELS,
        target='Annotated ictal epochs, not preictal prediction; seizure-free controls at least 300 seconds from any annotated event'), issues)


def lanl_audit(root, spec):
    url = 'https://raw.githubusercontent.com/ebpfly/shmtools/747392d82d17e5ef8d1bbe31bbd696894151b0e5/examples/data/data_files/data3SS.mat'
    path = root / 'data3SS.mat'
    if not path.exists(): path.write_bytes(fetch(url))
    data = path.read_bytes()
    blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    if blob != spec['git_blob_sha1']: raise ValueError('pinned MATLAB blob mismatch')
    raw = loadmat(path)
    report = dict(source=url, sha256=file_sha256(path), git_blob_sha1=blob,
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in raw.items() if not k.startswith('__')},
        states=np.unique(raw['states']).tolist(), labels=raw['states'].reshape(-1).tolist(),
        warning='Mirror README sampling rate conflicts with original report. No SPI computation released before metadata review.')
    atomic_json(root / 'input-audit.json', report)
    print(json.dumps(report), flush=True)


def structural_split(seed=SEED):
    rng = np.random.default_rng(seed)
    linear = rng.permutation(np.arange(1, 10)).tolist()
    contact = rng.permutation(np.arange(10, 18)).tolist()
    return dict(source=linear[:5] + contact[:4], screen=linear[5:7] + contact[4:6],
                confirmation_reserved=linear[7:] + contact[6:], seed=seed)


def lanl_prepare(root, spec):
    """Small mirror subset, with condition semantics verified against LA-14393."""
    path = root / 'data3SS.mat'
    original_audit = json.loads((root / 'input-audit.json').read_text())
    if file_sha256(path) != original_audit['sha256']:
        raise ValueError('audited structural input changed')
    report_url = 'https://digital.library.unt.edu/ark:/67531/metadc930290/m2/1/high_res_d/961604.pdf'
    report = root / 'LA-14393.pdf'
    if not report.exists(): report.write_bytes(fetch(report_url))
    if file_sha256(report) != spec['report_sha256']:
        raise ValueError('reviewed technical report changed')
    source = loadmat(path)
    data, labels = source['dataset'], source['states'].reshape(-1)
    if data.shape != (8192, 5, 170) or not np.array_equal(np.unique(labels), np.arange(1,18)):
        raise ValueError('unexpected structural teaching subset')
    if not np.all(np.unique(labels, return_counts=True)[1] == 10):
        raise ValueError('unexpected structural repetition count')
    split = structural_split()
    atomic_json(root / 'split.json', split)
    rows, physical, force = [], {}, {}
    for role in ['source', 'screen']:
        for condition in split[role]:
            for repeat, index in enumerate(np.flatnonzero(labels == condition)):
                key = f'state{condition:02d}-repeat{repeat:02d}'
                # No decimation: the original excitation reaches 150 Hz.
                physical[key] = data[2048:6144, 1:, index].astype(np.float64)
                force[key] = data[2048:6144, 0, index]
                rows.append(dict(row_id=key, group=f'state{condition:02d}', role=role,
                    target=int(condition >= 10), condition=condition, repeat=repeat,
                    native_index=int(index), native_sha256=file_sha256(path)))
    atomic_json(root / 'mirror-audit.json', original_audit)
    atomic_savez(root / 'excitation-control.npz', force)
    finalize(root, 'lanl3ss', rows, physical, dict(dataset=original_audit['source'],
        data_sha256=file_sha256(path), report=report_url, report_sha256=file_sha256(report),
        report_evidence='LA-14393 printed pp. 7-9, Tables 1-2: channel 1 load cell, channels 2-5 accelerometers, 320 Hz, states 1-9 linear and 10-17 contact',
        sample_rate_hz=320, crop_samples=[2048,6144], channels=[2,3,4,5], split=split,
        independent_unit='physical configuration of one apparatus; ten repetitions are not independent structures',
        confirmation_scope='reserved configurations of the same apparatus, not cross-structure validation',
        excitation_control_sha256=file_sha256(root / 'excitation-control.npz'),
        limitation='Pinned third-party copy with ten repetitions/condition, matching the report count; original file/trial identities and stored calibrated units are not independently certified.'), [])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['chbmit', 'hmc', 'grabmyo', 'lanl3ss'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release-lanl', action='store_true', help='Prepare the audited structural subset after report verification')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / 'manifest.json').exists(): raise FileExistsError('prepared corpus already exists')
    spec = yaml.safe_load(PROTOCOL.read_text())['datasets'][args.dataset]
    if args.release_lanl and args.dataset != 'lanl3ss': raise ValueError('LANL release flag for wrong dataset')
    dict(chbmit=chbmit, hmc=hmc, grabmyo=grabmyo,
         lanl3ss=lanl_prepare if args.release_lanl else lanl_audit)[args.dataset](args.root, spec)
