"""Prepare character-axis corpora on Gadi: linear surrogates and three EEG cohorts.

Cohorts share one window contract: the 19 standard 10-20 channels in a fixed
order, each cohort's own reference (re-referencing to their average would make
every conditional statistic singular), 0.5-45 Hz zero-phase band-pass, 100 Hz,
per-channel standardization, windows inside contiguous data only. Missing
canonical channels exclude a participant; nothing is interpolated, because
interpolation creates linear dependence between channels. Absolute band power
is kept before standardization as an amplitude-informed baseline.

Raw files are streamed into memory and discarded after windowing; their URL,
size and SHA-256 are recorded, so no raw copies consume project inodes.
"""
from __future__ import annotations

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
import time
import urllib.request
import zipfile

import numpy as np
from scipy import signal
from scipy.io import loadmat

from mtsspi.provenance import atomic_json, atomic_savez
from mtsspi.surrogates import multivariate_iaaft
from research.representation.empirical_transfer.prepare import write_corpus

SEED = 261102
FS = 100
CANON = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
ALIAS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
# HydroCel GSN 128 equivalents of the 10-20 positions (EGI technical note).
EGI = dict(Fp1='E22', Fp2='E9', F7='E33', F3='E24', Fz='E11', F4='E124', F8='E122', T7='E45', C3='E36', Cz='Cz',
           C4='E104', T8='E108', P7='E58', P3='E52', Pz='E62', P4='E92', P8='E96', O1='E70', O2='E83')
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
OPENNEURO = 'https://s3.amazonaws.com/openneuro.org'
LEMON = 'https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON'
PROPOFOL = 'https://www.repository.cam.ac.uk/bitstreams/e94a6722-da5b-4e53-8673-5e8ec106e0f7/download'


def fetch(url, attempts=5):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'mts-spi-research'})
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def receipt(url, data):
    return dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def eeglab(set_bytes, fdt_bytes=None):
    """Return (data[channel, sample, trial], labels, srate, events) from EEGLAB bytes."""
    m = loadmat(io.BytesIO(set_bytes), squeeze_me=True, struct_as_record=False)
    E = m['EEG'] if 'EEG' in m else None
    get = (lambda k: getattr(E, k)) if E is not None else (lambda k: m[k])
    nb, pnts, trials = int(get('nbchan')), int(get('pnts')), int(get('trials'))
    data = get('data')
    if isinstance(data, str):
        if fdt_bytes is None:
            raise ValueError('EEGLAB data stored in a separate .fdt file')
        data = np.frombuffer(fdt_bytes, '<f4').reshape((nb, pnts * trials), order='F')
    data = np.asarray(data, dtype=np.float64).reshape((nb, pnts, trials), order='F')
    labels = [str(c.labels) for c in np.atleast_1d(get('chanlocs'))]
    events = [(str(e.type), float(e.latency)) for e in np.atleast_1d(get('event'))] if np.size(get('event')) else []
    return data, labels, float(get('srate')), events


def condition(x, fs):
    """(channel, sample) native -> (sample, channel) at FS, band-passed 0.5-45 Hz."""
    sos = signal.butter(4, [.5, 45], btype='bandpass', fs=fs, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=1)
    ratio = Fraction(FS / fs).limit_denominator(1000)
    if ratio != 1:
        y = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=1)
    return y.T


def select(labels, mapping=None):
    mapping = mapping or {c: c for c in CANON}
    names = {ALIAS.get(l, l): k for k, l in enumerate(labels)}
    missing = [c for c in CANON if mapping[c] not in names]
    return ([] if missing else [names[mapping[c]] for c in CANON]), missing


def windows(segments, length, count, margin=100):
    starts = [s for a, b in segments for s in range(a + margin, b - margin - length + 1, length)]
    if len(starts) < count:
        return starts
    return [starts[k] for k in np.unique(np.round(np.linspace(0, len(starts) - 1, count)).astype(int))]


def band_power(w):
    f, p = signal.welch(w, fs=FS, nperseg=200, axis=0)
    return np.log10(np.stack([p[(f >= a) & (f < b)].mean(0) for a, b in BANDS]).T.ravel())


def cut(y, starts, length, key, group, target, role, extra):
    rows, arrays, power = [], {}, {}
    for k, s in enumerate(starts):
        w = y[s:s + length]
        sd = w.std(0)
        if not np.isfinite(w).all() or np.any(sd < 1e-9):
            continue
        name = f'{key}-w{k:02d}'
        power[name] = band_power(w)
        arrays[name] = ((w - w.mean(0)) / sd).astype(np.float64)
        rows.append(dict(row_id=name, group=group, target=int(target), role=role, start_sample=int(s), **extra))
    return rows, arrays, power


def finalize(root, name, rows, arrays, power, provenance, issues):
    atomic_savez(root / 'absolute-power.npz', dict(row_id=np.array([r['row_id'] for r in rows]),
                 power=np.stack([power[r['row_id']] for r in rows]),
                 feature=np.array([f'{c}:{a}-{b}Hz' for c in CANON for a, b in BANDS])))
    provenance.update(channels=CANON, sample_rate_hz=FS, filter_hz=[.5, 45], issues=issues)
    write_corpus(root, name, rows, arrays, provenance)


def ad_ftd(root, count=8, length=2000):
    base = f'{OPENNEURO}/ds004504'
    table = fetch(f'{base}/participants.tsv')
    people = list(csv.DictReader(io.StringIO(table.decode()), delimiter='\t'))
    code = dict(C=0, A=1, F=2)

    def task(p):
        sid = p['participant_id']
        url = f'{base}/derivatives/{sid}/eeg/{sid}_task-eyesclosed_eeg.set'
        data = fetch(url)
        x, labels, fs, _ = eeglab(data)
        index, missing = select(labels)
        if missing:
            return [], {}, {}, [dict(subject=sid, missing=missing)], receipt(url, data)
        y = condition(x[index, :, 0], fs)
        starts = windows([(10 * FS, len(y) - 10 * FS)], length, count)
        r, a, w = cut(y, starts, length, sid, sid, code[p['Group']], 'cohort',
                      dict(age=int(p['Age']), sex=p['Gender'], mmse=int(p['MMSE']), diagnosis=p['Group']))
        return r, a, w, [] if len(r) == count else [dict(subject=sid, windows=len(r))], receipt(url, data)

    rows, arrays, power, issues, receipts = [], {}, {}, [], []
    with ThreadPoolExecutor(8) as pool:
        for r, a, w, i, rec in pool.map(task, people):
            rows += r; arrays.update(a); power.update(w); issues += i; receipts.append(rec)
    finalize(root, 'ad-ftd-r10-261102', rows, arrays, power, dict(dataset='OpenNeuro ds004504 derivatives (ASR + ICA cleaned, A1-A2 reference)',
             participants_sha256=hashlib.sha256(table).hexdigest(), sources=receipts,
             target='0 control, 1 Alzheimer disease, 2 frontotemporal dementia; independent unit = participant'), issues)


def lemon(root, count=6, length=2000):
    meta_url = f'{LEMON}/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    meta = fetch(meta_url)
    people = {r['ID']: r for r in csv.DictReader(io.StringIO(meta.decode()))}
    listing = fetch(f'{LEMON}/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/').decode()
    available = sorted(set(re.findall(r'href="(sub-\d+)_EC\.set"', listing)))
    young, old = {'20-25', '25-30', '30-35'}, {'55-60', '60-65', '65-70', '70-75', '75-80'}

    def task(sid):
        age = people.get(sid, {}).get('Age')
        if age not in young | old:
            return [], {}, {}, [dict(subject=sid, excluded=f'age bin {age}')], None
        base = f'{LEMON}/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/{sid}_EC'
        s, f = fetch(base + '.set'), fetch(base + '.fdt')
        x, labels, fs, events = eeglab(s, f)
        index, missing = select(labels)
        rec = [receipt(base + '.set', s), receipt(base + '.fdt', f)]
        if missing:
            return [], {}, {}, [dict(subject=sid, missing=missing)], rec
        y = condition(x[index, :, 0], fs)
        cuts = sorted({0, len(y)} | {int(round((lat - 1) * FS / fs)) for t, lat in events if t == 'boundary'})
        starts = windows(list(zip(cuts[:-1], cuts[1:])), length, count)
        r, a, w = cut(y, starts, length, sid, sid, int(age in old), 'cohort',
                      dict(age_bin=age, sex=people[sid]['Gender_ 1=female_2=male']))
        return r, a, w, [] if len(r) == count else [dict(subject=sid, windows=len(r))], rec

    rows, arrays, power, issues, receipts = [], {}, {}, [], []
    with ThreadPoolExecutor(8) as pool:
        for r, a, w, i, rec in pool.map(task, available):
            rows += r; arrays.update(a); power.update(w); issues += i; receipts += rec or []
    finalize(root, 'lemon-age-r10-261102', rows, arrays, power, dict(dataset='MPI-LEMON preprocessed eyes-closed EEG (ICA cleaned, common average reference)',
             meta=receipt(meta_url, meta), sources=receipts,
             target='0 aged 20-35, 1 aged 55-80 (5-year bins); independent unit = participant'), issues)


def propofol(root, count=10, inspect=False):
    archive = Path(os.environ.get('PBS_JOBFS', root)) / 'sedation-restingstate.zip'
    if not archive.exists():
        data = fetch(PROPOFOL)
        archive.write_bytes(data)
        atomic_json(root / 'archive-receipt.json', receipt(PROPOFOL, data)); del data
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        info_name = next(n for n in members if n.endswith('datainfo.mat'))
        info = loadmat(io.BytesIO(z.read(info_name)), squeeze_me=True)['datainfo']
        table = [dict(name=str(r[0]), level=int(r[1]), plasma=float(r[2]), rt=float(r[3]), correct=float(r[4])) for r in info]
        sets = {Path(n).stem: n for n in members if n.endswith('.set')}
        fdts = {Path(n).stem: n for n in members if n.endswith('.fdt')}
        if inspect:
            first = table[0]['name']
            x, labels, fs, events = eeglab(z.read(sets[first]), z.read(fdts[first]) if first in fdts else None)
            atomic_json(root / 'inspection.json', dict(members=members[:400], datainfo=table, srate=fs, shape=list(x.shape),
                        labels=labels, egi_missing=select(labels, EGI)[1], canonical_missing=select(labels)[1]))
            return
        rows, arrays, power, issues = [], {}, {}, []
        for entry in table:
            x, labels, fs, _ = eeglab(z.read(sets[entry['name']]), z.read(fdts[entry['name']]) if entry['name'] in fdts else None)
            mapping = EGI if select(labels, EGI)[1] == [] else None
            index, missing = select(labels, mapping)
            if missing:
                issues.append(dict(dataset=entry['name'], missing=missing)); continue
            subject = entry['name'][:2]
            epochs = x.shape[2]
            chosen = np.unique(np.round(np.linspace(0, epochs - 1, min(count, epochs))).astype(int))
            for e in chosen:
                y = condition(x[index, :, e], fs)
                r, a, w = cut(y, [0], len(y), f'{subject}-level{entry["level"]}-e{e:03d}', subject, entry['level'] - 1, 'cohort',
                              dict(level=entry['level'], plasma=entry['plasma'], rt=entry['rt'], correct=entry['correct'], epoch=int(e)))
                rows += r; arrays.update(a); power.update(w)
    finalize(root, 'propofol-r10-261102', rows, arrays, power, dict(dataset='Chennu et al. 2016 propofol sedation, Cambridge Apollo 1810/252736',
             archive=json.loads((root / 'archive-receipt.json').read_text()),
             target='sedation level - 1 (0 baseline, 1 mild, 2 moderate, 3 recovery); independent unit = participant'), issues)


def surrogates(root, bank_path, name):
    bank = np.load(bank_path)
    rows, arrays, diagnostics = [], {}, []
    for k, (row, x) in enumerate(zip(bank['row_id'].astype(str), bank['x'])):
        s, info = multivariate_iaaft(np.asarray(x, dtype=np.float64), np.random.default_rng([SEED, k]))
        s = (s - s.mean(0)) / s.std(0)
        key = f's-{row}'
        arrays[key] = s
        rows.append(dict(row_id=key, source_row=row, group=str(bank['group'][k]), target=int(bank['y'][k]), role=str(bank['role'][k])))
        diagnostics.append(dict(row_id=key, **info))
    write_corpus(root, f'surrogate-{name}-r10-261102', rows, arrays,
                 dict(source_bank=str(bank_path), generator='multivariate IAAFT, one surrogate per record, seed [261102, index]',
                      maximum_amplitude_relative_error=max(d['amplitude_relative_error'] for d in diagnostics),
                      maximum_correlation_error=max(d['correlation_max_abs_error'] for d in diagnostics)))
    atomic_json(root / 'surrogate-diagnostics.json', diagnostics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', choices=['ad_ftd', 'lemon', 'propofol', 'surrogate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / 'manifest.json').exists():
        raise FileExistsError('prepared corpus already exists')
    if args.corpus == 'surrogate':
        surrogates(args.root, args.bank, args.name)
    elif args.corpus == 'propofol':
        propofol(args.root, inspect=args.inspect)
    else:
        dict(ad_ftd=ad_ftd, lemon=lemon)[args.corpus](args.root)
