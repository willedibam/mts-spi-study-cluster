"""Arm-by-arm table for the low-rate MI pair (preregs/lowrate-prereg.md), as review/sumall.py and montage/animal_level.py.

    python summarise.py <scores dir with one folder per arm> <100 Hz zns dir> [arm ...]

Per arm, normal-scored (the anchor) and raw: residual = z_real - z_surr per window; per session Δ = state - awake_ec
mean; eyes-open session SD (the arm's own); per agent mean Δ, in own SD, sessions down; four animal means over drug
sessions and one-sided t(3); the same animals' eyes-open; awake real z; K < 0 fractions and Theiler cap hits by state.
For arms on the bank's own 10-s windows, also r between each window's residual and its 100 Hz residual.
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

AGENTS = ('KT', 'KTMD', 'MD', 'PF', 'SLP')


def load(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/zns_[AB]_*.npz'))]
    cat = lambda k: np.concatenate([x[k] for x in L])
    out = {k: cat(k) for k in ('V', 'id', 'unit', 'state', 'agent', 'animal')}
    out['E'] = cat('E') if 'E' in L[0].files else None
    out['D'] = cat('D') if 'D' in L[0].files else None
    out['T'] = int(L[0]['T']) if 'T' in L[0].files else 1000
    return out


def sessions(x, v):
    res = x['V'][:, v, 0] - x['V'][:, v, 1]; unit, state, agent, animal = x['unit'], x['state'], x['agent'], x['animal']
    S = {}
    for u in np.unique(unit):
        m = lambda s: (unit == u) & (state == s)
        if not m('awake_ec').any():
            continue
        b = res[m('awake_ec')].mean()
        S[u] = dict(ag=agent[unit == u][0], an=animal[unit == u][0],
                    **{s: res[m(s)].mean() - b for s in ('drug', 'sleep', 'awake_eo', 'recovery_ec') if m(s).any()})
    return S


def table(x, name, ref=None):
    lines = []
    for v, vn in ((1, 'ns'), (0, 'raw')):
        S = sessions(x, v); eo = np.array([S[u]['awake_eo'] for u in S if 'awake_eo' in S[u]]); sd = eo.std(ddof=1)
        cells = []
        for ag in AGENTS:
            s = 'sleep' if ag == 'SLP' else 'drug'; a = np.array([S[u][s] for u in S if S[u]['ag'] == ag and s in S[u]])
            cells.append(f'{ag} {a.mean():+.4f} ({a.mean() / sd:+.1f} SD, {np.sum(a < 0)}/{len(a)})')
        A = {}; E_ = {}
        for u in S:
            if S[u]['ag'] != 'SLP' and 'drug' in S[u]:
                A.setdefault(S[u]['an'], []).append(S[u]['drug']); E_.setdefault(S[u]['an'], []).append(S[u].get('awake_eo', np.nan))
        am = np.array([np.mean(A[k]) for k in sorted(A)]); t = ttest_1samp(am, 0, alternative='less')
        eom = np.nanmean([np.nanmean(E_[k]) for k in sorted(A)])
        lines.append(f'{name:6s} {vn:3s} EO SD {sd:.4f} (EO mean {eo.mean():+.4f}) | ' + ' | '.join(cells) +
                     f' | animals {", ".join(f"{k} {np.mean(A[k]):+.4f}" for k in sorted(A))}; t(3) {t.statistic:.2f}, p {t.pvalue:.4f};'
                     f' pooled {am.mean():+.4f} = {am.mean() / sd:+.1f} SD; EO per-animal {eom:+.4f}')
    z = x['V'][:, 1, :]; st = x['state']
    lvl = '; '.join(f'{s} z {z[st == s, 0].mean():.4f}/{z[st == s, 1].mean():.4f}' for s in ('awake_ec', 'drug', 'sleep'))
    lines.append(f'{name:6s} T {x["T"]} n_win {len(st)} | ns real/surr {lvl}')
    if x['E'] is not None:
        K = x['E'][:, 1, :, 1]   # normal-scored; [window, real/surr, edge]
        lines.append(f'{name:6s} K<0 (ns real/surr) ' + '; '.join(
            f'{s} {np.mean(K[st == s, 0] < 0):.3f}/{np.mean(K[st == s, 1] < 0):.3f}' for s in ('awake_ec', 'awake_eo', 'drug', 'sleep')))
    if x['D'] is not None:
        D = x['D'][:, 1]         # normal-scored; [window, real/surr, (capped fraction, median w, cap)]
        lines.append(f'{name:6s} Theiler cap {int(D[0, 0, 2])}: capped fraction (ns real/surr), median w ' + '; '.join(
            f'{s} {D[st == s, 0, 0].mean():.3f}/{D[st == s, 1, 0].mean():.3f}, {np.median(D[st == s, 0, 1]):.0f}' for s in ('awake_ec', 'drug', 'sleep')))
    if ref is not None and len(x['id']) == len(ref['id']) and (x['id'] == ref['id']).all():
        r0 = ref['V'][:, 1, 0] - ref['V'][:, 1, 1]; r1 = x['V'][:, 1, 0] - x['V'][:, 1, 1]
        lines.append(f'{name:6s} window residual r with 100 Hz: {np.corrcoef(r0, r1)[0, 1]:.2f}')
    return '\n'.join(lines)


if __name__ == '__main__':
    root, z100 = Path(sys.argv[1]), sys.argv[2]
    ref = load(z100)
    arms = sys.argv[3:] or sorted(p.name for p in root.iterdir() if p.is_dir() and glob.glob(f'{p}/zns_*.npz'))
    print(table(ref, '100Hz'))
    for a in arms:
        print(table(load(root / a), a, ref))
