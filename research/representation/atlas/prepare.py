"""Window banks for the consciousness atlas: per-dataset loaders, one preprocessing chain, the bank format `extract.py` reads.

Arms (see `design.md`):
- long: 10-s windows at 100 Hz (T = 1000). The whole recording is notched at mains (source rate), high-passed at
  0.5 Hz, low-passed at 40 Hz (zero-phase Butterworth) and resampled to 100 Hz by polyphase filtering. Windows
  do not overlap, are spread evenly over each labelled interval after trimming EDGE seconds at both ends, pass
  fixed artefact rules and are capped per unit-state.
- short: 2-s windows (T = 200), identical for epoched and continuous sources. Each 2.25-s epoch at source rate
  (the fly's released epochs; for continuous sources, centred on each selected long window) is detrended,
  high-passed at 1 Hz, low-passed at 40 Hz and resampled on its own, and its central 2 s are kept.

Reference: common average over the unit's good channels, then the last good channel is dropped, because
average-referenced data sum to zero and their covariance has rank M - 1 (the precision SPIs would then be
pseudo-inverse artefacts). Bipolar releases (fly) are kept as released. The reference is applied after
filtering, which is the same thing because both are linear and act identically on every channel.

    python -m research.representation.atlas.prepare neurotycho --root $MTS_SCRATCH/data/atlas --out BANKS --workers 8
    python -m research.representation.atlas.prepare fly --root $MTS_SCRATCH/data/atlas --out BANKS
    python -m research.representation.atlas.prepare subset BANK --ids ID [ID ...] --out SMALL_BANK
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, detrend, iirnotch, resample_poly, sosfiltfilt, filtfilt, welch

FS = 100
LONG, SHORT, EPOCH, EDGE = 10.0, 2.0, 2.25, 5.0
FLAT_SD, NOISY_SD, ROBUST_SD, OUTLIER_FRACTION = 0.01, 5.0, 6.0, 0.05


# ---------------------------------------------------------------- filters
def lowpass(x, fs, cutoff=40.0, order=8):
    return sosfiltfilt(butter(order, cutoff, fs=fs, output='sos'), x, axis=0)


def highpass(x, fs, cutoff, order=2):
    return sosfiltfilt(butter(order, cutoff, btype='highpass', fs=fs, output='sos'), x, axis=0)


def notch(x, fs, mains):
    if mains is None or mains >= fs / 2:
        return x
    b, a = iirnotch(mains, 30.0, fs=fs)
    return filtfilt(b, a, x, axis=0)


def to_rate(x, fs):
    ratio = Fraction(FS) / Fraction(fs).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def continuous(x, fs, mains):
    """Source-rate recording (n, M) -> 100 Hz, for the long arm."""
    return to_rate(lowpass(highpass(notch(x, fs, mains), fs, 0.5), fs), fs)


def short_epoch(x, fs):
    """One 2.25-s epoch (n, M) at source rate -> its central 2 s at 100 Hz, shape (200, M)."""
    y = to_rate(lowpass(highpass(detrend(x, axis=0), fs, 1.0), fs), fs)
    n = int(round(SHORT * FS))
    start = (len(y) - n) // 2
    if start < 0:
        raise ValueError(f'epoch shorter than {SHORT} s')
    return y[start:start + n]


def reference(x, rule):
    """'car': common average over the given channels, then drop the last; 'asis': unchanged."""
    if rule == 'asis':
        return x
    if rule != 'car':
        raise ValueError(rule)
    return (x - x.mean(axis=1, keepdims=True))[:, :-1]


# ---------------------------------------------------------------- screening and selection
def flat_runs(x, fs):
    """Boolean (n,) marking samples inside a run of at least fs identical consecutive values in any channel."""
    flat = np.zeros(len(x), bool)
    for column in np.atleast_2d(x.T):
        same = np.flatnonzero(np.diff(column) != 0)
        edges = np.concatenate([[-1], same, [len(column) - 1]])
        for a, b in zip(edges[:-1] + 1, edges[1:] + 1):
            if b - a >= fs:
                flat[a:b] = True
    return flat


def bad_channels(x):
    """Indices of flat (SD < 1% of median) or noisy (SD > 5x median) or non-finite channels in (n, M) data."""
    sd = np.nanstd(x, axis=0)
    median = np.median(sd[np.isfinite(sd)])
    return sorted(set(np.flatnonzero(~np.isfinite(x).all(axis=0)).tolist()) |
                  set(np.flatnonzero((sd < FLAT_SD * median) | (sd > NOISY_SD * median)).tolist()))


def robust_scale(x):
    centre = np.median(x, axis=0)
    return centre, 1.4826 * np.median(np.abs(x - centre), axis=0)


def window_ok(w, centre, scale, flat):
    if not np.isfinite(w).all() or flat.any():
        return False
    outlying = np.abs(w - centre) > ROBUST_SD * scale
    return bool((outlying.mean(axis=0) <= OUTLIER_FRACTION).all())


def spread(n_good, cap):
    """Indices of at most cap items spread evenly over n_good candidates."""
    if n_good <= cap:
        return list(range(n_good))
    return sorted(set(np.round(np.linspace(0, n_good - 1, cap)).astype(int).tolist()))


def delta_fraction(w):
    f, p = welch(w, fs=FS, nperseg=min(len(w), 4 * FS), axis=0)
    p = p.sum(axis=1)
    return float(p[(f >= 0.5) & (f < 4)].sum() / p[(f >= 0.5) & (f <= 40)].sum())


def conditioning(w):
    z = (w - w.mean(axis=0)) / w.std(axis=0)
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    return int(np.sum(eig > eig.max() * len(eig) * np.finfo(float).eps)), float(eig.max() / max(eig.min(), 1e-300))


# ---------------------------------------------------------------- NeuroTycho
NT_STATES = {'AwakeEyesClosed': 'awake_ec', 'AwakeEyesOpened': 'awake_eo', 'Anesthetized': 'drug',
             'RecoveryEyesClosed': 'recovery_ec', 'Sleeping': 'sleep'}
NT_CAPS = dict(awake_ec=10, awake_eo=10, drug=10, recovery_ec=10, sleep=30)
NT_FS, NT_MAINS = 1000, 50.0


def mat_vector(raw):
    data = loadmat(io.BytesIO(raw))
    [key] = [k for k in data if not k.startswith('__')]
    return np.ravel(data[key]).astype(np.float64)


def nt_intervals(raw):
    """{state: [(start, end) in source samples]} from a Condition.mat (one-based ConditionIndex)."""
    c = loadmat(io.BytesIO(raw), simplify_cells=True)
    labels, index = np.atleast_1d(c['ConditionLabel']), np.atleast_1d(c['ConditionIndex']).astype(int) - 1
    starts, out = {}, {}
    for label, i in zip(labels, index):
        m = re.fullmatch(r'(\w+)-(Start|End)', str(label))
        if not m or m[1] not in NT_STATES:
            continue
        if m[2] == 'Start':
            starts[m[1]] = i
        elif m[1] in starts:
            out.setdefault(NT_STATES[m[1]], []).append((starts.pop(m[1]), int(i)))
    return out


def nt_archive(path):
    """One NeuroTycho archive -> long and short windows with rows, and the unit record."""
    path = Path(path)
    unit = path.stem
    L = int(LONG * FS)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        sessions = sorted({m[1] for n in names if (m := re.fullmatch(r'(.*/Session\d+)/Condition\.mat', n))})
        info = sorted(n for n in names if re.search(r'/Info-[^/]*\.txt$', n))
        intervals = {s: nt_intervals(z.read(f'{s}/Condition.mat')) for s in sessions}
        used = [s for s in sessions if intervals[s]]
        channels = sorted(int(m[1]) for n in names if (m := re.fullmatch(re.escape(used[0]) + r'/ECoG_ch(\d+)\.mat', n)))
        if channels != list(range(1, len(channels) + 1)):
            raise ValueError(f'{unit}: channel files are not 1..M')
        load = lambda s, ch: mat_vector(z.read(f'{s}/ECoG_ch{ch}.mat'))
        # Pass 1: every channel at 100 Hz, and per-channel raw flat-run or non-finite masks, per session.
        low, flat = {}, {}
        for s in used:
            columns, masks = [], []
            for ch in channels:
                x = load(s, ch)[:, None]
                columns.append(continuous(np.nan_to_num(x), NT_FS, NT_MAINS)[:, 0])
                f = flat_runs(x, NT_FS) | ~np.isfinite(x[:, 0])
                n = len(columns[-1])
                masks.append(np.pad(f, (0, max(0, n * 10 - len(f))))[:n * 10].reshape(n, 10).any(axis=1))
            low[s], flat[s] = np.column_stack(columns), np.column_stack(masks)
        spans = {state: [(s, a // 10 + int(EDGE * FS), b // 10 - int(EDGE * FS)) for s in used
                         for a, b in intervals[s].get(state, [])] for state in NT_CAPS}
        labelled = np.concatenate([low[s][a:b] for v in spans.values() for s, a, b in v if b > a])
        bad = bad_channels(labelled)
        good = [i for i in range(len(channels)) if i not in bad]
        flat = {s: m[:, good].any(axis=1) for s, m in flat.items()}
        # Selection per state over all of that state's intervals; artefacts are judged against that state's own spread.
        selected = []
        for state, cap in NT_CAPS.items():
            if not any(b - a >= L for _, a, b in spans[state]):
                continue
            centre, scale = robust_scale(reference(np.concatenate([low[s][a:b, good] for s, a, b in spans[state] if b > a]), 'car'))
            candidates = [(s, k) for s, a, b in spans[state] for k in range(a, b - L + 1, L)
                          if window_ok(reference(low[s][k:k + L, good], 'car'), centre, scale, flat[s][k:k + L])]
            selected += [(state, *candidates[i]) for i in spread(len(candidates), cap)]
        long_rows, long_arrays, count = [], {}, {}
        for state, s, k in selected:
            w = reference(low[s][k:k + L, good], 'car')
            rank, cond = conditioning(w)
            count[state] = count.get(state, -1) + 1
            wid = f'{unit}:{state}:{count[state]:02d}'
            long_arrays[wid] = w
            long_rows.append(dict(id=wid, unit=unit, state=state, session=s.rsplit('/', 1)[1], start_s=k / FS,
                                  M=w.shape[1], delta_fraction=delta_fraction(w), rank=rank, condition=cond))
        del low
        # Pass 2: notched raw 2.25-s epochs centred on each long window for the short arm; EMG and EOG where present.
        full = {s.rsplit('/', 1)[1]: s for s in used}
        raw = {s: np.column_stack([notch(load(full[s], channels[i]), NT_FS, NT_MAINS) for i in good])
               for s in sorted({r['session'] for r in long_rows})}
        extra = {}
        for s in raw:
            for name, lo, hi in (('EMG_Chin', 10.0, 200.0), ('EOG_REye', 0.3, 10.0)):
                member = f'{full[s]}/{name}.mat'
                if member in names:
                    v = mat_vector(z.read(member))
                    if len(v) == len(raw[s]):
                        extra[(s, name)] = sosfiltfilt(butter(2, [lo, hi], btype='bandpass', fs=NT_FS, output='sos'), v)
        half = int(EPOCH * NT_FS) // 2
        short_rows, short_arrays = [], {}
        for row in long_rows:
            s, a = row['session'], int(round(row['start_s'] * NT_FS))
            middle = a + int(LONG * NT_FS) // 2
            w = reference(short_epoch(raw[s][middle - half:middle + half], NT_FS), 'car')
            for name in ('EMG_Chin', 'EOG_REye'):
                if (s, name) in extra:
                    v = extra[(s, name)][a:a + int(LONG * NT_FS)]
                    row[name.split('_')[0].lower() + '_rms'] = float(np.sqrt(np.mean(v ** 2)))
            short_arrays[row['id']] = w
            short_rows.append(dict(id=row['id'], unit=unit, state=row['state'], session=s,
                                   start_s=row['start_s'] + (LONG - EPOCH) / 2, M=w.shape[1], rank=conditioning(w)[0]))
        meta = dict(unit=unit, info=z.read(info[0]).decode(errors='replace') if info else None,
                    sessions=[s.rsplit('/', 1)[1] for s in sessions], n_channels=len(channels),
                    bad_channels=[channels[i] for i in bad], dropped_after_car=channels[good[-1]],
                    kept_channels=[channels[i] for i in good[:-1]],
                    windows={state: count.get(state, -1) + 1 for state in NT_CAPS})
    return long_rows, long_arrays, short_rows, short_arrays, meta


def neurotycho(root, out, workers):
    raw = root / 'neurotycho'
    acquisition = json.loads((raw / 'acquisition.json').read_text())
    items = {i['path']: i for i in acquisition['files']}
    archives = sorted((raw / 'raw').glob('*.zip'))
    if len(archives) != len(items):
        raise ValueError(f'{len(archives)} archives on disk, {len(items)} in acquisition.json')
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(nt_archive, archives))
    for arm, T, li, ai in (('long', int(LONG * FS), 0, 1), ('short', int(SHORT * FS), 2, 3)):
        rows, arrays = [], {}
        for result, path in zip(results, archives):
            item = items[path.name]
            for r in result[li]:
                rows.append(dict(r, animal=item['animal'], agent=item['agent'], date=item['date']))
            arrays.update(result[ai])
        write_bank(out / f'neurotycho-{arm}', 'neurotycho', arm, T, rows, arrays,
                   dict(units=[r[4] for r in results], reference='car, last good channel dropped', mains=NT_MAINS,
                        caps=NT_CAPS, source=str(raw)))


# ---------------------------------------------------------------- fly (Leung et al. 2025, figshare 12309950)
def fly(root, out):
    path = next((root / 'fly13' / 'raw').glob('split2250_*.mat'))
    data = loadmat(path)['fly_data']  # samples x channels x epochs x flies x (wake, anaesthetised) [figshare description]
    if data.shape[:2] != (2250, 15) or data.shape[4] != 2:
        raise ValueError(f'unexpected fly array {data.shape}')
    rows, arrays = [], {}
    for f in range(data.shape[3]):
        for c, state in enumerate(('awake', 'isoflurane')):
            for e in range(data.shape[2]):
                x = np.asarray(data[:, :, e, f, c], dtype=np.float64)
                if not np.isfinite(x).all():
                    continue
                w = short_epoch(x, 1000)
                wid = f'fly{f + 1:02d}:{state}:{e:02d}'
                arrays[wid] = w
                rows.append(dict(id=wid, unit=f'fly{f + 1:02d}', state=state, epoch=e, M=w.shape[1], rank=conditioning(w)[0]))
    write_bank(out / 'fly-short', 'fly13', 'short', int(SHORT * FS), rows, arrays,
               dict(reference='bipolar as released', source=str(path), source_sha256=sha256(path)))


# ---------------------------------------------------------------- bank format
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write_bank(bank, dataset, arm, T, rows, arrays, record):
    if len({r['id'] for r in rows}) != len(rows) or set(arrays) != {r['id'] for r in rows}:
        raise ValueError('window ids are not unique or do not match the arrays')
    if any(a.shape != (T, r['M']) for r, a in ((r, arrays[r['id']]) for r in rows)):
        raise ValueError('a window has the wrong shape')
    bank.mkdir(parents=True, exist_ok=True)
    partial = bank / 'windows.partial.npz'
    np.savez(partial, **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})
    partial.replace(bank / 'windows.npz')
    manifest = dict(dataset=dataset, arm=arm, fs=FS, T=T, rows=rows, preprocessing=record,
                    prepare_sha256=sha256(Path(__file__)))
    (bank / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{bank}: {len(rows)} windows, M {sorted({r["M"] for r in rows})}', flush=True)


def subset(bank, ids, out):
    manifest = json.loads((bank / 'manifest.json').read_text())
    rows = [r for r in manifest['rows'] if r['id'] in set(ids)]
    if len(rows) != len(set(ids)):
        raise ValueError('some ids are not in the bank')
    with np.load(bank / 'windows.npz') as archive:
        arrays = {r['id']: archive[r['id']] for r in rows}
    write_bank(out, manifest['dataset'], manifest['arm'], manifest['T'], rows, arrays,
               dict(manifest['preprocessing'], subset_of=str(bank)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=['neurotycho', 'fly', 'subset'])
    parser.add_argument('bank', type=Path, nargs='?')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--ids', nargs='+')
    a = parser.parse_args()
    if a.dataset == 'neurotycho':
        neurotycho(a.root, a.out, a.workers)
    elif a.dataset == 'fly':
        fly(a.root, a.out)
    else:
        subset(a.bank, a.ids, a.out)
