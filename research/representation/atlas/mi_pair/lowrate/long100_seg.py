"""Addendum A to preregs/long100-prereg.md: the W50 windows against a 10-s segment null.

Does the 50-s change rest on slow change between the 10-s pieces of a window (drift in gain or linear coupling), which a
stationary IAAFT over 50 s cannot reproduce? Per W50 window, as lowrate/score.py's normal-scored branch: the same rows,
32-channel subsets and normal-scoring; the null is nonstat's N2 (joint IAAFT, spectrum-final, run separately on each
1000-sample piece and concatenated), seed default_rng([row, 1000]) as nonstat/run.py. The real window's KSG is not
recomputed: z_real comes from the W50 score, matched by id, subset labels and the Gaussian-MI vector (alignment gate).

  score:  python long100_seg.py score <W50 bank dir> <out.npz> <agent> <first|row> <n workers>
  calib:  python long100_seg.py calib <W50 stage-a bank> <out.npz> <n block> <n gain_shared> <n workers>
  cread:  python long100_seg.py cread <calib.npz> <calib_long100.npz>
  read:   python long100_seg.py read <scores dir with W50/ and W50seg/>

calib: long100.py's generators, sources and seeds at T = 5000, scored against N0 (reproduces calib_long100's dz: gate)
and N2. Size: the drug-blind offset mean(KTMD, PF) - AW of z_real - z_N2 on `block`.
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / 'nonstat')]
import edges as E                                     # noqa: E402
from nulls import n2                                  # noqa: E402
from mtsspi.surrogates import multivariate_iaaft      # noqa: E402

SEG = 1000                                            # 10 s at 100 Hz
AGENTS = ('KT', 'KTMD', 'MD', 'PF', 'SLP')


def gauss(v):                                         # edges.edge_mi's Gaussian MI, without the KSG
    x = (v - v.mean(0)) / v.std(0); r = np.corrcoef(x.T)[np.triu_indices(x.shape[1], 1)]
    return -0.5 * np.log(1 - r ** 2)


def one(args):
    x, seed = args; v = E.ns(x)
    gs, ks = E.edge_mi(n2(v, np.random.default_rng([seed, SEG]), SEG))
    return np.array([gs, ks], np.float32), gauss(v).astype(np.float32)


def score(d, out, agent, rowmode, nw):              # rows, subsets and seeds as lowrate/score.py
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in E.STATES and r['agent'] in agent.split(',')]
    S0 = E.sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else E.sub32(r['M'])
    with Pool(nw) as p:
        res = p.map(one, [(W[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, kept = E.meta(rows, man)
    labels = np.array([np.array(kept[r['unit']][:r['M']])[sub(r)] for r in rows])
    np.savez(out, Eseg=np.array([a for a, _ in res]), Greal=np.array([b for _, b in res]), chan=labels, rowmode=rowmode,
             T=man['T'], seg=SEG, **m)
    print('done', out, flush=True)


def calib_one(a):
    import long100 as LH
    kind, seed, src = a; x = E.ns(LH.gen(kind, src, np.random.default_rng(seed)))
    s0 = multivariate_iaaft(x, np.random.default_rng(seed + 1), final='spectrum')[0]
    s2 = n2(x, np.random.default_rng([seed, SEG]), SEG)
    return [E.zc(*E.edge_mi(v)) for v in (x, s0, s2)]


def calib(d, out, nb, ng, nw):                       # sources and seeds as long100.calib, T = 5000 only
    import long100 as LH
    man = json.loads((Path(d) / 'manifest.json').read_text()); W = np.load(Path(d) / 'windows.npz'); src = {}
    for tag, ag, st in (('PF', 'PF', 'drug'), ('KTMD', 'KTMD', 'drug'), ('AW', 'KTMD', 'awake_ec')):
        rows = [r for r in man['rows'] if r['state'] in E.STATES and r['agent'] == ag]; S0 = E.sub32(rows[0]['M'])
        src[tag] = [E.ns(W[r['id']].astype(float)[:, S0]) for r in rows if r['state'] == st]
    jobs, lab = [], []
    for si, tag in enumerate(('AW', 'KTMD', 'PF')):
        for ki, (kind, n) in enumerate((('block', nb), ('gain_shared', ng))):
            for i in range(n):
                jobs.append((kind, 8 * 10 ** 6 + 10 ** 5 * si + 10 ** 4 * ki + 2 * i, src[tag][i % len(src[tag])][:LH.L]))
                lab.append((kind, tag, i))
    with Pool(nw) as p:
        Z = p.map(calib_one, jobs, chunksize=1)
    np.savez(out, z=np.array(Z), kind=np.array([k for k, _, _ in lab]), tag=np.array([t for _, t, _ in lab]),
             i=np.array([i for _, _, i in lab]))
    print('done', out, flush=True)


def cread(f, ref):
    """Frozen size reading (Addendum A): the N2 offset passes if |offset| <= 2 SE or |offset| <= .001."""
    D, R = np.load(f), np.load(ref); z, kind, tag, idx = D['z'], D['kind'], D['tag'], D['i']
    gap = []
    for k in ('block', 'gain_shared'):                # gate: z_real - z_N0 reproduces long100 calib's dz (T = 5000)
        for t in ('AW', 'KTMD', 'PF'):
            r = R['dz'][(R['kind'] == k) & (R['tag'] == t) & (R['T'] == 5000)]; m = (kind == k) & (tag == t)
            gap.append(np.max(np.abs((z[m, 0] - z[m, 1]) - r[idx[m]])))
    print(f'gate: max |z_real - z_N0 - calib_long100 dz| = {max(gap):.2e} ->', 'PASS' if max(gap) < 1e-8 else 'FAIL')
    off = {}
    for j, nm in ((1, 'N0'), (2, 'N2')):
        m_ = {}
        for t in ('AW', 'KTMD', 'PF'):
            b = z[(kind == 'block') & (tag == t), 0] - z[(kind == 'block') & (tag == t), j]
            g = z[(kind == 'gain_shared') & (tag == t), 0] - z[(kind == 'gain_shared') & (tag == t), j]
            m_[t] = (b.mean(), b.std(ddof=1) / np.sqrt(len(b)))
            print(f'{nm} {t:4s} block {b.mean():+.4f} +- {m_[t][1]:.4f} (n {len(b)}) | gain_shared {g.mean():+.4f} (n {len(g)})')
        o = np.mean([m_['KTMD'][0], m_['PF'][0]]) - m_['AW'][0]
        se = np.sqrt((m_['KTMD'][1] ** 2 + m_['PF'][1] ** 2) / 4 + m_['AW'][1] ** 2); off[nm] = (o, se)
        print(f'{nm} drug-blind offset {o:+.4f} +- {se:.4f}')
    o, se = off['N2']
    print('N2 size:', 'PASS' if abs(o) <= 2 * se or abs(o) <= .001 else 'FAIL')
    gk = lambda j: np.mean(z[kind == 'gain_shared', 0] - z[kind == 'gain_shared', j])
    print(f'gain_shared retained under N2 / N0: {gk(2) / gk(1):.2f} (descriptive: a 5-s-dwell gain switch)')


def load(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/zns_[AB]_*.npz'))]
    return L, (lambda k: np.concatenate([x[k] for x in L]))


def animals(res, unit, state, agent, animal):         # summarise.sessions / animal_drug on a residual vector
    S = {}
    for u in np.unique(unit):
        m = lambda s: (unit == u) & (state == s)
        if m('awake_ec').any():
            b = res[m('awake_ec')].mean()
            S[u] = dict(ag=agent[unit == u][0], an=animal[unit == u][0],
                        **{s: res[m(s)].mean() - b for s in ('drug', 'sleep', 'awake_eo') if m(s).any()})
    A = {}
    for u in S:
        if S[u]['ag'] != 'SLP' and 'drug' in S[u]:
            A.setdefault(S[u]['an'], []).append(S[u]['drug'])
    eo = np.array([S[u]['awake_eo'] for u in S if 'awake_eo' in S[u]])
    ag = {a: np.mean([S[u]['sleep' if a == 'SLP' else 'drug'] for u in S if S[u]['ag'] == a and ('sleep' if a == 'SLP' else 'drug') in S[u]])
          for a in AGENTS if any(S[u]['ag'] == a for u in S)}
    return {k: np.mean(A[k]) for k in sorted(A)}, eo.std(ddof=1), ag


def read(root):
    """Frozen reading (Addendum A). R_seg = pooled drug delta (real - N2) / pooled drug delta (real - N0) over animal
    means, on W50, normal-scored. R_seg is read only if W50's own N0 delta is <= -2 of its eyes-open SD with 4/4 animals
    negative; otherwise only the differences are reported. Not drift: R_seg >= .75; mostly drift: R_seg <= .5; else
    partial. Each call is conditional on the N2 size check (cread) passing."""
    Lw, cw = load(Path(root) / 'W50'); Ls, cs = load(Path(root) / 'W50seg')
    idw, ids = cw('id'), cs('id'); pos = {k: i for i, k in enumerate(idw)}
    assert len(set(ids)) == len(ids) and set(ids) <= set(pos), 'W50seg ids not a subset of W50'
    j = np.array([pos[k] for k in ids]); Ew = cw('E')[:, 1].astype(float)
    assert (cw('chan')[j] == cs('chan')).all(), 'subset labels differ'
    gap = np.max(np.abs(Ew[j, 0, 0] - cs('Greal'))); print(f'alignment gate: max |G_real(W50) - G_real(seg)| = {gap:.2e}')
    assert gap < 1e-5, 'alignment gate failed'
    zr, z0 = cw('V')[j, 1, 0], cw('V')[j, 1, 1]; Es = cs('Eseg').astype(float)
    z2 = np.array([np.corrcoef(g, k)[0, 1] for g, k in Es])
    meta = [cs(k) for k in ('unit', 'state', 'agent', 'animal')]
    (a0, sd0, g0), (a2, sd2, g2) = animals(zr - z0, *meta), animals(zr - z2, *meta)
    k = sorted(a0); p0, p2 = np.mean([a0[i] for i in k]), np.mean([a2[i] for i in k]); d = np.array([a2[i] - a0[i] for i in k])
    print(f'W50 real - N0: pooled drug {p0:+.4f} ({p0 / sd0:+.1f} own EO SD {sd0:.4f}); animals ' + ', '.join(f'{i} {a0[i]:+.4f}' for i in k))
    print(f'W50 real - N2: pooled drug {p2:+.4f} ({p2 / sd2:+.1f} own EO SD {sd2:.4f}); animals ' + ', '.join(f'{i} {a2[i]:+.4f}' for i in k))
    print('agents N0 | N2: ' + '  '.join(f'{a} {g0[a]:+.4f} | {g2.get(a, np.nan):+.4f}' for a in g0))
    print(f'N2 - N0 per animal: {np.round(d, 4)}, t(3) {ttest_1samp(d, 0).statistic:+.2f}')
    if p0 <= -2 * sd0 and all(a0[i] < 0 for i in k):
        R = p2 / p0; call = 'not drift' if R >= .75 else 'mostly drift' if R <= .5 else 'partial'
        print(f'R_seg {R:.2f} -> {call} (conditional on the N2 size check)')
    else:
        print('W50 N0 delta below the floor: R_seg not read; differences only')


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'score':
        score(a[2], a[3], a[4], a[5], int(a[6]))
    elif a[1] == 'calib':
        calib(a[2], a[3], int(a[4]), int(a[5]), int(a[6]))
    elif a[1] == 'cread':
        cread(a[2], a[3])
    else:
        read(a[2])
