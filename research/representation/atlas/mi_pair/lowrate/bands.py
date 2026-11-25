"""Frequency-band banks for the macaque MI pair at 100 Hz (preregs/bands-prereg.md): which band carries the drug fall.

Same frame as the anchor: the continuous 0.5-40 Hz, 100 Hz signal rebuilt from the 1 kHz archives (pass 1 of
prepare.nt_archive, as lowrate/build.py), the bank's own good channels and 10-s windows (T = 1000), CAR per window
after filtering. Only the band changes. Filters are order-8 zero-phase Butterworth on the continuous signal (no window
edge effects): prepare.lowpass, prepare.highpass(order=8), and band-pass as both.
  LP<f>     low-pass at f Hz       0.5-f      (LP4, LP8 are lowrate's L10, L20 and are not rebuilt)
  HP<f>     high-pass at f Hz      f-40
  BP<a>-<b> band-pass              a-b
Gate A as build.py: the rebuilt unfiltered windows must equal the stage banks' windows, or nothing is written.

    PYTHONPATH=<repo>:<repo>/src python research/representation/atlas/mi_pair/lowrate/bands.py <banks dir> <archives dir> <out dir> <workers>
Writes <out>/<arm>/stage-{a,b}/{manifest.json,windows.npz} (bank format) and <out>/gate_a_bands.json.
"""
import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.representation.atlas import prepare as P

ARMS = {'LP2': (None, 2), 'LP15': (None, 15), 'LP25': (None, 25),
        'HP2': (2, None), 'HP4': (4, None), 'HP8': (8, None), 'HP15': (15, None),
        'BP2-4': (2, 4), 'BP4-8': (4, 8), 'BP8-15': (8, 15)}
ORDER = 8
T = int(P.LONG * P.FS)                                # 1000 samples
STAGES = ('stage-a', 'stage-b')


def band(x, lo, hi):
    if hi is not None:
        x = P.lowpass(x, P.FS, hi, ORDER)
    if lo is not None:
        x = P.highpass(x, P.FS, lo, ORDER)
    return x


def archive(args):
    path, rows, unit_meta = args
    with zipfile.ZipFile(path) as z:                  # pass 1, as prepare.nt_archive and lowrate/build.py
        names = set(z.namelist())
        sessions = sorted({m[1] for n in names if (m := re.fullmatch(r'(.*/Session\d+)/Condition\.mat', n))})
        used = [s for s in sessions if P.nt_intervals(z.read(f'{s}/Condition.mat'))]
        channels = sorted(int(m[1]) for n in names if (m := re.fullmatch(re.escape(used[0]) + r'/ECoG_ch(\d+)\.mat', n)))
        assert channels == list(range(1, len(channels) + 1))
        good = [c - 1 for c in unit_meta['kept_channels']] + [unit_meta['dropped_after_car'] - 1]
        low = {s.rsplit('/', 1)[1]: np.column_stack([P.continuous(np.nan_to_num(P.mat_vector(z.read(f'{s}/ECoG_ch{ch}.mat'))[:, None]),
                                                                  P.NT_FS, P.NT_MAINS)[:, 0] for ch in channels])[:, good]
               for s in used}
    start = lambda row: int(round(row['start_s'] * P.FS))
    gate, banks = [], {}
    for row in rows:
        s, k = row['session'], start(row)
        w = P.reference(low[s][k:k + T], 'car')
        if row['_bank'] not in banks:
            banks[row['_bank']] = np.load(row['_bank'])
        b = banks[row['_bank']][row['id']]
        gate.append(float(np.max(np.abs(w - b)) / b.std()) if w.shape == b.shape else np.inf)
    for f in banks.values():
        f.close()
    out = {a: {} for a in ARMS}
    for s in low:                                     # one filtered copy at a time (memory)
        for a in ARMS:
            y = band(low[s], *ARMS[a])
            for row in rows:
                if row['session'] == s:
                    out[a][row['id']] = P.reference(y[start(row):start(row) + T], 'car')
    return Path(path).stem, gate, out


if __name__ == '__main__':
    banks, archives, out, nw = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    man = {g: json.loads((banks / f'neurotycho-long-{g}' / 'manifest.json').read_text()) for g in STAGES}
    units = {u['unit']: u for u in man['stage-a']['preprocessing']['units']}
    rows_of = {}
    for g in STAGES:
        for r in man[g]['rows']:
            rows_of.setdefault(r['unit'], []).append(dict(r, _bank=str(banks / f'neurotycho-long-{g}' / 'windows.npz')))
    with ProcessPoolExecutor(nw) as pool:
        res = list(pool.map(archive, [(archives / f'{u}.zip', rows_of[u], units[u]) for u in sorted(rows_of)]))
    gate = {u: max(g) for u, g, _ in res}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gate_a_bands.json').write_text(json.dumps(dict(max_rel_diff=gate, worst=max(gate.values())), indent=1) + '\n')
    print('gate A worst max|rebuilt - bank| / SD:', max(gate.values()), flush=True)
    if max(gate.values()) > 1e-9:
        sys.exit('gate A failed: rebuilt 100 Hz windows differ from the bank')
    sha = P.sha256(Path(__file__))
    for a, (lo, hi) in ARMS.items():
        arrays = {k: v for _, _, o in res for k, v in o[a].items()}
        for g in STAGES:
            rows, bank = man[g]['rows'], out / a / g
            bank.mkdir(parents=True, exist_ok=True)
            assert all(arrays[r['id']].shape == (T, r['M']) for r in rows)
            np.savez(bank / 'windows.partial.npz', **{r['id']: np.asarray(arrays[r['id']], np.float64) for r in rows})
            (bank / 'windows.partial.npz').replace(bank / 'windows.npz')
            rec = dict(man[g]['preprocessing'], arm=a, band_hz=[lo or 0.5, hi or 40], filter_order=ORDER,
                       filter='order-8 zero-phase Butterworth (prepare.lowpass / prepare.highpass) on the continuous 100 Hz signal',
                       window_s=P.LONG, build_sha256=sha, source_bank=f'neurotycho-long-{g}')
            (bank / 'manifest.json').write_text(json.dumps(dict(dataset='neurotycho', arm=a, fs=P.FS, T=T, rows=rows,
                                                                preprocessing=rec), indent=1) + '\n')
        print(a, {g: len(man[g]['rows']) for g in STAGES}, flush=True)
    print('ALLDONE', flush=True)
