"""Low-rate window banks for the macaque MI pair (preregs/lowrate-prereg.md), rebuilt from the 1 kHz NeuroTycho archives.

Per archive, pass 1 of prepare.nt_archive is repeated (every channel to the continuous 0.5-40 Hz, 100 Hz signal) and
the bank's own good channels and window starts are reused (manifest `kept_channels` + `dropped_after_car`, `session`,
`start_s`), so no selection is redone for the 10-s arms. Gate A: the rebuilt 100 Hz windows must equal the stage
banks' windows, or nothing is written.

Arms (r = 20 or 10 Hz, q = 100 / r; LP_r = prepare.lowpass at 0.8 x r/2 Hz, order-8 zero-phase Butterworth, applied
to the continuous 100 Hz signal, so no window edge effects; CAR per window after it, as prepare):
  D<r>   anti-aliased: LP_r, then every q-th sample of the bank's 10-s window      T = 1000 / q
  S<r>   plain subsampling: every q-th sample of the bank's 10-s window, no filter  T = 1000 / q
  L<r>   LP_r kept at 100 Hz, the bank's 10-s window                               T = 1000
  D<r>L  anti-aliased, sample-matched: q x 10 s windows, every q-th sample          T = 1000
  S<r>L  plain subsampling of the same long windows                               T = 1000
Long windows are selected as prepare selects 10-s ones (same spans, EDGE trim, state-own artefact scale at 100 Hz,
flat-run mask, non-overlapping, spread evenly, NT_CAPS), with length q x 10 s. Subsampling offset is 0 (the window's
first sample).

    PYTHONPATH=<repo>:<repo>/src python research/representation/atlas/mi_pair/lowrate/build.py <banks dir> <archives dir> <out dir> <workers>
Writes <out>/<arm>/stage-{a,b}/{manifest.json,windows.npz} in the bank format edges.py reads, and <out>/gate_a.json.
"""
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re

import numpy as np

from research.representation.atlas import prepare as P

RATES = {20: 5, 10: 10}                               # rate (Hz) -> decimation factor at 100 Hz
CUT = {r: 0.8 * r / 2 for r in RATES}                 # 8 and 4 Hz
L10 = int(P.LONG * P.FS)                              # 1000 samples
STAGES = ('stage-a', 'stage-b')


def ref(x):
    return P.reference(x, 'car')


def archive(args):
    path, rows, unit_meta = args
    path, unit = Path(path), Path(path).stem
    with zipfile.ZipFile(path) as z:                  # pass 1, as prepare.nt_archive
        names = set(z.namelist())
        sessions = sorted({m[1] for n in names if (m := re.fullmatch(r'(.*/Session\d+)/Condition\.mat', n))})
        intervals = {s: P.nt_intervals(z.read(f'{s}/Condition.mat')) for s in sessions}
        used = [s for s in sessions if intervals[s]]
        channels = sorted(int(m[1]) for n in names if (m := re.fullmatch(re.escape(used[0]) + r'/ECoG_ch(\d+)\.mat', n)))
        assert channels == list(range(1, len(channels) + 1))
        good = [c - 1 for c in unit_meta['kept_channels']] + [unit_meta['dropped_after_car'] - 1]
        low, flat = {}, {}
        for s in used:
            columns, masks = [], []
            for ch in channels:
                x = P.mat_vector(z.read(f'{s}/ECoG_ch{ch}.mat'))[:, None]
                columns.append(P.continuous(np.nan_to_num(x), P.NT_FS, P.NT_MAINS)[:, 0])
                f = P.flat_runs(x, P.NT_FS) | ~np.isfinite(x[:, 0])
                n = len(columns[-1])
                masks.append(np.pad(f, (0, max(0, n * 10 - len(f))))[:n * 10].reshape(n, 10).any(axis=1))
            low[s] = np.column_stack(columns)[:, good]
            flat[s] = np.column_stack(masks)[:, good].any(axis=1)
    full = {s.rsplit('/', 1)[1]: s for s in used}
    lp = {(s, r): P.lowpass(low[s], P.FS, CUT[r]) for s in used for r in RATES}
    out = {a: {} for a in [f'{k}{r}' for r in RATES for k in 'DSL'] + [f'{k}{r}L' for r in RATES for k in 'DS']}
    gate, banks = [], {}
    for row in rows:                                  # the bank's 10-s windows
        s, k = full[row['session']], int(round(row['start_s'] * P.FS))
        w = ref(low[s][k:k + L10])
        if row['_bank'] not in banks:
            banks[row['_bank']] = np.load(row['_bank'])
        b = banks[row['_bank']][row['id']]
        gate.append(float(np.max(np.abs(w - b)) / b.std()) if w.shape == b.shape else np.inf)
        for r, q in RATES.items():
            out[f'D{r}'][row['id']] = ref(lp[s, r][k:k + L10:q])
            out[f'S{r}'][row['id']] = ref(low[s][k:k + L10:q])
            out[f'L{r}'][row['id']] = ref(lp[s, r][k:k + L10])
    long_rows = {r: [] for r in RATES}
    spans = {state: [(s, a // 10 + int(P.EDGE * P.FS), b // 10 - int(P.EDGE * P.FS)) for s in used
                     for a, b in intervals[s].get(state, [])] for state in P.NT_CAPS}
    info = {k: rows[0][k] for k in ('animal', 'agent', 'date')}
    for r, q in RATES.items():                        # sample-matched long windows, selected as prepare does
        L = L10 * q
        for state, cap in P.NT_CAPS.items():
            if not any(b - a >= L for _, a, b in spans[state]):
                continue
            centre, scale = P.robust_scale(ref(np.concatenate([low[s][a:b] for s, a, b in spans[state] if b > a])))
            cand = [(s, k) for s, a, b in spans[state] for k in range(a, b - L + 1, L)
                    if P.window_ok(ref(low[s][k:k + L]), centre, scale, flat[s][k:k + L])]
            for c, i in enumerate(P.spread(len(cand), cap)):
                s, k = cand[i]
                wid = f'{unit}:{state}:{c:02d}'
                out[f'D{r}L'][wid] = ref(lp[s, r][k:k + L:q])
                out[f'S{r}L'][wid] = ref(low[s][k:k + L:q])
                long_rows[r].append(dict(id=wid, unit=unit, state=state, session=s.rsplit('/', 1)[1], start_s=k / P.FS,
                                         M=len(good) - 1, delta_fraction=P.delta_fraction(ref(low[s][k:k + L])),
                                         n_candidates=len(cand), **info))
    for f in banks.values():
        f.close()
    return unit, gate, out, long_rows


def write(bank, rows, arrays, T, fs, record):
    bank.mkdir(parents=True, exist_ok=True)
    assert {r['id'] for r in rows} <= set(arrays) and all(   # arrays hold both stages; this bank gets its own rows
        arrays[r['id']].shape == (T, r['M']) for r in rows)
    np.savez(bank / 'windows.partial.npz', **{r['id']: np.asarray(arrays[r['id']], np.float64) for r in rows})
    (bank / 'windows.partial.npz').replace(bank / 'windows.npz')
    (bank / 'manifest.json').write_text(json.dumps(dict(dataset='neurotycho', arm=record['arm'], fs=fs, T=T, rows=rows,
                                                        preprocessing=record), indent=1) + '\n')


if __name__ == '__main__':
    banks, archives, out, nw = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    man = {g: json.loads((banks / f'neurotycho-long-{g}' / 'manifest.json').read_text()) for g in STAGES}
    units = {u['unit']: u for u in man['stage-a']['preprocessing']['units']}
    stage_of, rows_of = {}, {}
    for g in STAGES:
        for r in man[g]['rows']:
            stage_of[r['unit']] = g
            rows_of.setdefault(r['unit'], []).append(dict(r, _bank=str(banks / f'neurotycho-long-{g}' / 'windows.npz')))
    jobs = [(archives / f'{u}.zip', rows_of[u], units[u]) for u in sorted(rows_of)]
    with ProcessPoolExecutor(nw) as pool:
        res = list(pool.map(archive, jobs))
    gate = {u: max(g) for u, g, _, _ in res}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gate_a.json').write_text(json.dumps(dict(max_rel_diff=gate, worst=max(gate.values())), indent=1) + '\n')
    print('gate A worst max|rebuilt - bank| / SD:', max(gate.values()), flush=True)
    if max(gate.values()) > 1e-9:
        sys.exit('gate A failed: rebuilt 100 Hz windows differ from the bank')
    arrays = {a: {} for a in res[0][2]}
    for _, _, o, _ in res:
        for a in o:
            arrays[a].update(o[a])
    long_rows = {r: [x for *_, lr in res for x in lr[r]] for r in RATES}
    sha = P.sha256(Path(__file__))
    for a in arrays:
        r = int(re.search(r'\d+', a)[0]); q = RATES[r]
        for g in STAGES:
            if a.endswith('L'):
                rows = [x for x in long_rows[r] if stage_of[x['unit']] == g]
            else:
                rows = man[g]['rows']
            T, fs = (L10, P.FS) if a[0] == 'L' else ((L10, r) if a.endswith('L') else (L10 // q, r))
            rec = dict(man[g]['preprocessing'], arm=a, rate_hz=r, lowpass_hz=CUT[r] if a[0] in 'DL' else None,
                       lowpass='prepare.lowpass (order-8 Butterworth, zero-phase) on the continuous 100 Hz signal',
                       window_s=(L10 * q if a.endswith('L') else L10) / P.FS, subsample_offset=0,
                       build_sha256=sha, source_bank=f'neurotycho-long-{g}')
            write(out / a / g, rows, arrays[a], T, fs, rec)
        print(a, {g: len([1 for x in (long_rows[r] if a.endswith('L') else man[g]['rows'])
                          if not a.endswith('L') or stage_of[x['unit']] == g]) for g in STAGES}, flush=True)
    print('ALLDONE', flush=True)
