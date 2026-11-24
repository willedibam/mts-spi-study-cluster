"""Window extraction for HUP ds004100 interictal runs (rules: ../preregs/hup-prereg.md). No MI here.

    python extract.py <ds004100 dir> <out dir> [subject ...]

Per subject saves <out>/<subject>.npz: W (windows, 1000, M) float32 after CAR (µV, one non-SOZ non-resected channel
dropped), rows (JSON per window: run, onset_s, burst, delta_fraction), spike (windows, M) bool: any sample beyond
5 robust SDs of the run's scale, meta (JSON: names, soz, resect, xyz in mm or null, implant, engel, fs, dropped).
Writes <out>/<subject>.qc.json (one record; separate files so array jobs do not race).
"""
import sys, os, glob, json, re
import numpy as np, pandas as pd, mne
from scipy.signal import detrend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from research.representation.atlas.prepare import continuous, bad_channels, robust_scale, window_ok, flat_runs, delta_fraction
mne.set_log_level('ERROR')
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0)))
MIN_SOZ_SAVED, EDGE_S, WIN_S, SPIKE_SD = 4, 10.0, 10.0, 5.0


def tsv(p):
    return pd.read_csv(p, sep='\t', encoding='utf-8-sig', keep_default_na=False)


def subject(root, sub, out, part):
    qc = []
    runs = sorted(glob.glob(f'{root}/{sub}/ses-*/ieeg/*task-interictal*_ieeg.edf'))
    if not runs: return [dict(subject=sub, excluded='no interictal run')]
    chs = [tsv(f.replace('_ieeg.edf', '_channels.tsv')) for f in runs]
    good = [n for n in chs[0]['name'] if all(((c['name'] == n) & (c['status'] == 'good')).any() for c in chs)]
    desc = dict(zip(chs[0]['name'], chs[0]['status_description'].astype(str)))
    if sum('soz' in desc[n] for n in good) < MIN_SOZ_SAVED:
        return [dict(subject=sub, excluded=f'< {MIN_SOZ_SAVED} good SOZ contacts in metadata')]
    el = glob.glob(f'{root}/{sub}/ses-*/ieeg/*_electrodes.tsv'); E = tsv(el[0]).set_index('name') if el else None
    xyz = {n: (None if E is None or n not in E.index or str(E.loc[n, 'x']) in ('n/a', '') else [float(E.loc[n, a]) for a in 'xyz'])
           for n in good}
    Y, F, fss = [], [], []
    for f in runs:
        raw = mne.io.read_raw_edf(f, preload=False); fs = raw.info['sfreq']; fss.append(fs)
        src = raw.get_data(picks=[raw.ch_names.index(n) for n in good]).T * 1e6
        fl = flat_runs(src, int(round(fs))); Y.append(continuous(src, fs, 60.0))
        F.append(fl[np.minimum((np.arange(len(Y[-1])) * fs / 100).astype(int), len(fl) - 1)])
    bad = set(bad_channels(np.concatenate(Y)))
    keep = [j for j in range(len(good)) if j not in bad]; names = [good[j] for j in keep]
    soz = ['soz' in desc[n] for n in names]; res = ['resect' in desc[n] and 'soz' not in desc[n] for n in names]
    drop = max(i for i in range(len(names)) if not soz[i] and not res[i])
    an = [i for i in range(len(names)) if i != drop]
    W, rows, SP = [], [], []; rec = dict(subject=sub, M_good=len(good), bad=[good[j] for j in sorted(bad)], dropped=names[drop],
                                          M=len(an), n_soz=int(sum(soz[i] for i in an)), runs={})
    for r, (y, fl) in enumerate(zip(Y, F)):
        y = y[:, keep]; c = (y - y.mean(1, keepdims=True))[:, an]; centre, scale = robust_scale(c); n_ok = 0
        starts = np.arange(EDGE_S, len(c) / 100 - EDGE_S - WIN_S + 1e-9, WIN_S)
        for s in starts:
            i = int(round(s * 100)); w = c[i:i + 1000]
            if not window_ok(w, centre, scale, fl[i:i + 1000]): continue
            W.append(detrend(w, axis=0).astype(np.float32)); SP.append((np.abs(w - centre) > SPIKE_SD * scale).any(0)); n_ok += 1
            rows.append(dict(run=r + 1, onset_s=float(s), burst=burst(w), delta_fraction=delta_fraction(w)))
        rec['runs'][r + 1] = dict(candidates=len(starts), passed=n_ok)
    p = part.get(sub, {})
    meta = dict(names=[names[i] for i in an], soz=[soz[i] for i in an], resect=[res[i] for i in an],
                xyz=[xyz[names[i]] for i in an], implant=p.get('implant', ''), engel=p.get('engel', ''), fs=fss, dropped=names[drop])
    np.savez(f'{out}/{sub}.npz', W=np.array(W), spike=np.array(SP), rows=json.dumps(rows), meta=json.dumps(meta))
    qc.append(rec); print(json.dumps(rec), flush=True)
    return qc


if __name__ == '__main__':
    root, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    subs = sys.argv[3:] or sorted(os.path.basename(d) for d in glob.glob(f'{root}/sub-*'))
    part = {r['participant_id']: r for r in tsv(f'{root}/participants.tsv').to_dict('records')}
    for s in subs:
        json.dump(subject(root, s, out, part), open(f'{out}/{s}.qc.json', 'w'), indent=1)
