"""Window selection for the human replication (rules: ../preregs/human-prereg.md). No z, no state contrast here.

    python extract.py prios   <ds004370 dir> <out dir> [letters from A, B, F]   (S-A, S-B; F: no epoch cap; prereg)
    python extract.py ds004541 <ds004541 dir> <out dir>

Per subject, saves <out>/<subject>.npz: W (windows, T, good channels; pre-CAR, µV, float32), rows (JSON: state, onset_s,
delay_s since previous pulse for PRIOS, burst, delta_fraction), meta (JSON: channel names, good, bad), and appends to qc.json.
"""
import sys, json, glob, os
import numpy as np, pandas as pd, mne
from scipy.signal import detrend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from research.representation.atlas.prepare import (short_epoch, continuous, bad_channels, window_ok, spread,
                                                   delta_fraction)
mne.set_log_level('ERROR')
burst = lambda c: float(np.median(1.4826 * np.median(np.abs(c - np.median(c, 0)), 0) / c.std(0)))


def has_flat(x, fs):
    """True if any channel of (n, M) source-rate data has a run of ≥ fs identical consecutive values."""
    same = np.diff(x, axis=0) == 0
    for col in same.T:
        if not col.any(): continue
        e = np.flatnonzero(np.diff(np.concatenate([[0], col.view(np.int8), [0]])))
        if (e[1::2] - e[::2]).max() >= fs - 1: return True
    return False


def blocked(intervals, a, b):
    return any(s < b and e > a for s, e in intervals)


def finish(subject, cands, names, out, qc, cap, extra, own_scale=False):
    """cands: list of (state, onset_s, W100 (T, M) pre-CAR, flat, info). Bad-channel screen, awake scale, window_ok, spread.
    info['stim'] (PRIOS): names of the stimulated electrodes, which carry the stimulator's common signal between pulses
    (Deviation 1); they are left out of the CAR, the artefact check and the indices, and score.py leaves them out of z."""
    X = np.concatenate([c[2] for c in cands]); bad = bad_channels(X); good = [j for j in range(X.shape[1]) if j not in bad]
    gname = [names[j] for j in good]
    def car(c):  # CAR over the good, non-stimulated channels; stimulated ones NaN; last dropped. Also the kept mask.
        keep = np.array([n not in c[4].get('stim', ()) for n in gname]); w = c[2][:, good].astype(float)
        r = w - w[:, keep].mean(1, keepdims=True); r[:, ~keep] = np.nan; return r[:, :-1], keep[:-1]
    def robust(state):  # robust_scale, NaN-aware
        x = np.concatenate([car(c)[0] for c in cands if c[0] == state]); m = np.nanmedian(x, 0)
        return m, 1.4826 * np.nanmedian(np.abs(x - m), 0)
    centre, scale = robust('awake')
    W, rows, rec = [], [], dict(subject=subject, M=len(names), bad=[names[j] for j in bad], M_good=len(good), **extra)
    for state in ('awake', 'anaes'):
        cs = [c for c in cands if c[0] == state]
        if own_scale and cs: centre, scale = robust(state)  # S-B
        ok = [c for c in cs if not c[3] and (lambda r, k: window_ok(r[:, k], centre[k], scale[k], np.zeros(1, bool)))(*car(c))]
        sel = [ok[i] for i in spread(len(ok), cap)]
        for c in sel:
            w = c[2][:, good]; r, k = car(c); r = r[:, k]; W.append(w.astype(np.float32))
            info = dict(c[4]); info.pop('stim', None)
            if 'stim' in c[4]: info['stim'] = [j for j, n in enumerate(gname) if n in c[4]['stim']]
            rows.append(dict(state=state, onset_s=c[1], burst=burst(r), delta_fraction=delta_fraction(r), **info))
        rec[state] = dict(cand=len(cs), passed=len(ok), used=len(sel))
    np.savez(f'{out}/{subject}.npz', W=np.array(W), rows=json.dumps(rows), meta=json.dumps(dict(names=names, good=good, bad=bad)))
    qc.append(rec); print(json.dumps(rec), flush=True)


def prios(root, out, qc, cap=100, opt=''):
    for sdir in sorted(glob.glob(f'{root}/sub-PRIOS*')):
        subject = os.path.basename(sdir); runs = sorted(glob.glob(f'{sdir}/ses-1/ieeg/*_ieeg.vhdr'))
        ch = [pd.read_csv(r.replace('_ieeg.vhdr', '_channels.tsv'), sep='\t') for r in runs]
        keep = set.intersection(*[set(c[(c['type'] == 'ECOG') & (c['status_description'] == 'included')]['name']) for c in ch])
        names = [n for n in ch[0]['name'] if n in keep]; cands = []
        for run in runs:
            state = 'anaes' if 'task-SPESprop' in run else 'awake'
            raw = mne.io.read_raw_brainvision(run, preload=False); fs = raw.info['sfreq']; picks = [raw.ch_names.index(n) for n in names]
            ev = pd.read_csv(run.replace('_ieeg.vhdr', '_events.tsv'), sep='\t')
            st = ev[ev['trial_type'] == 'electrical_stimulation'].assign(t=lambda e: e['onset'].astype(float)).sort_values('t')
            pulses = st['t'].values; sites = [str(s).split('-') for s in st['electrical_stimulation_site']]
            excl, local = [], []  # local: S-A, (start, end, electrodes) of artefacts naming electrodes
            for o, d, t, e0, e1 in zip(ev['onset'].astype(float), pd.to_numeric(ev['duration'], errors='coerce').fillna(0), ev['trial_type'],
                                       ev.get('electrodes_involved_onset', 'all'), ev.get('electrodes_involved_offset', 'all')):
                if t not in ('artefact', 'seizure', 'burst suppression'): continue
                el = {n.strip() for n in f'{e0},{e1}'.split(',')} - {'nan', ''}
                if 'A' in opt and t == 'artefact' and 'all' not in {x.lower() for x in el}: local.append((o - 2, o + d + 2, el))
                else: excl.append((o - 2, o + d + 2))
            for i, (p0, p1) in enumerate(zip(pulses[:-1], pulses[1:])):
                if p1 - p0 < 4.0: continue
                b = p1 - 0.25; a = b - 2.25
                if blocked(excl, a, b): continue
                s0 = int(round(a * fs)); n = int(round(2.25 * fs))
                if s0 + n > raw.n_times: continue
                x = raw.get_data(picks=picks, start=s0, stop=s0 + n).T * 1e6
                drop = set(sites[i] + sites[i + 1]).union(*[el for s, e, el in local if s < b and e > a])
                info = dict(delay_s=float(a - p0), run=os.path.basename(run), stim=sorted(drop))
                cands.append((state, a, short_epoch(x, fs), has_flat(x, fs), info))
        finish(subject, cands, names, out, qc, 10 ** 9 if 'F' in opt else cap, {}, own_scale='B' in opt)


def ds004541(root, out, qc, cap=30):
    for sdir in sorted(glob.glob(f'{root}/sub-*')):
        subject = os.path.basename(sdir); f = glob.glob(f'{sdir}/ses-01/eeg/*_eeg.edf')[0]
        ev = pd.read_csv(f.replace('_eeg.edf', '_events.tsv'), sep='\t', encoding='utf-8-sig'); t = ev['trial_type'].astype(str)
        on = lambda k: ev[t == k]['onset'].astype(float).values
        if not (len(on('loc')) and len(on('roc')) and len(on('start'))):
            qc.append(dict(subject=subject, excluded='missing loc, roc or start')); print(subject, 'excluded', flush=True); continue
        ch = pd.read_csv(f.replace('_eeg.edf', '_channels.tsv'), sep='\t', encoding='utf-8-sig')
        names = list(ch[(ch['type'] == 'EEG') & (ch['status'] != 'bad')]['name'])
        raw = mne.io.read_raw_edf(f, preload=False); fs = raw.info['sfreq']
        src = raw.get_data(picks=[raw.ch_names.index(n) for n in names]).T * 1e6; x = continuous(src, fs, 50.0); T_end = len(x) / 100
        stims = ev[t.str.startswith(('verbal', 'motor', 'tetanic', 'other'))]['onset'].astype(float).values
        excl = [(s - 15, s + 15) for s in stims]
        spans = dict(awake=(10.0, on('start')[0] - 10.0), anaes=(on('loc')[0] + 60.0, on('roc')[0] - 60.0)); cands = []
        for state, (a, b) in spans.items():
            for s in np.arange(a, min(b, T_end) - 10.0 + 1e-9, 10.0):
                if blocked(excl, s, s + 10.0): continue
                i = int(round(s * 100)); w = detrend(x[i:i + 1000], axis=0)
                j = int(round(s * fs)); cands.append((state, float(s), w, has_flat(src[j:j + int(10 * fs)], fs), {}))
        finish(subject, cands, names, out, qc, cap, dict(loc_s=float(on('loc')[0]), roc_s=float(on('roc')[0])))


if __name__ == '__main__':
    mode, root, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); qc = []
    prios(root, out, qc, opt=sys.argv[4] if len(sys.argv) > 4 else '') if mode == 'prios' else ds004541(root, out, qc)
    json.dump(qc, open(f'{out}/qc.json', 'w'), indent=1)
