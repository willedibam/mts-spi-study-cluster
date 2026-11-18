"""Frozen scorer for edges-prereg.md, Part E. Usage: python edges_score.py <dir with edges_*_*of*.npz> <out.json>

Fixed here before any output is read (thresholds the prereg left open are marked FIXED):
- E3 naming: the edges with the most negative held-out Δp (bottom 1% on the selection half) are called
  "gain" (a) if the mean fraction with ẽ above the median is > .60 in every primary animal, "fail to gain" (b)
  if < .40 in every primary animal, else "mixed" (FIXED).
- The p_s term uses real ĝ with surrogate k̂_s, as the prereg words it; k̂_s is the mean over three draws.
"""
import sys, glob, json, numpy as np
from scipy.stats import spearmanr

PRIMARY, TOP = ('KT', 'KTMD', 'PF'), .01


def load(d):
    rows = []
    for f in sorted(glob.glob(f'{d}/edges_*_*of*.npz')):
        z = np.load(f, allow_pickle=True)
        for a, unit, state, agent, animal, chan, row in zip(z['X'], z['unit'], z['state'], z['agent'], z['animal'], z['chan'], z['row']):
            rows.append(dict(X=np.asarray(a, float), unit=str(unit), state=str(state), agent=str(agent), animal=str(animal),
                             chan=tuple(json.loads(str(chan))), row=int(row), stage=f.split('edges_')[1][0]))
    rows.sort(key=lambda r: (r['stage'], r['row']))
    return rows


std = lambda v: (v - v.mean()) / v.std()


def session(R):
    """R: rows of one unit. Returns per-contrast half and full maps (ẽ, Δp) plus g for deciles."""
    st = np.array([r['state'] for r in R]); X = np.array([r['X'] for r in R])  # (w, 5, E): g, k, gs, ks, ksd
    e = X[:, 1] - X[:, 3]
    p = np.array([std(x[0]) * std(x[1]) for x in X]); ps = np.array([std(x[0]) * std(x[3]) for x in X]); pr = p - ps
    gw = X[st == 'awake_ec', 0].mean(0); dec = np.minimum((np.argsort(np.argsort(gw)) * 10) // len(gw), 9)
    adj = lambda d: d - np.array([d[dec == b].mean() for b in range(10)])[dec]
    tgt = 'sleep' if R[0]['agent'] == 'SLP' else 'drug'
    out = {}
    for name, s in (('drug', tgt), ('eo', 'awake_eo')):
        if not (st == s).any(): continue
        idx = lambda s_, h: np.flatnonzero(st == s_)[h::2] if h is not None else np.flatnonzero(st == s_)
        m = lambda a, h: a[idx(s, h)].mean(0) - a[idx('awake_ec', h)].mean(0)
        out[name] = dict(full=adj(m(e, None)), h=[adj(m(e, 0)), adj(m(e, 1))], dp=[m(pr, 0), m(pr, 1)], dpfull=m(pr, None),
                         zres=float(np.mean(m(pr, None))))
    return out, tgt


def conc(sel, meas, neg=False):
    s, v = (-sel, -meas) if neg else (sel, meas)
    top = np.argsort(s)[-max(1, int(np.ceil(TOP * len(s)))):]; pos = np.clip(v, 0, None)
    return float(pos[top].sum() / pos.sum()) if pos.sum() > 0 else np.nan


def held(c, f):
    return float(np.mean([f(c['h'][0], c['h'][1]), f(c['h'][1], c['h'][0])]))


def e3(c):
    rho = np.mean([spearmanr(c['dp'][a], c['h'][b])[0] for a, b in ((0, 1), (1, 0))])
    fr = []
    for a, b in ((0, 1), (1, 0)):
        bot = np.argsort(c['dp'][a])[:max(1, int(np.ceil(TOP * len(c['dp'][a]))))]
        fr.append(np.mean(c['h'][b][bot] > np.median(c['h'][b])))
    return float(rho), float(np.mean(fr))


def additive_r2(v, M):
    iu = np.triu_indices(M, 1); A = np.zeros((len(v), M)); A[np.arange(len(v)), iu[0]] = 1; A[np.arange(len(v)), iu[1]] = 1
    fit = A @ np.linalg.lstsq(A, v, rcond=None)[0]; return float(1 - np.sum((v - fit) ** 2) / np.sum((v - v.mean()) ** 2))


if __name__ == '__main__':
    rows = load(sys.argv[1]); units = sorted({r['unit'] + '|' + r['agent'] for r in rows}); S = {}
    for u in units:
        R = [r for r in rows if r['unit'] + '|' + r['agent'] == u]; c, tgt = session(R); M = len(R[0]['chan']); iu = np.triu_indices(M, 1)
        lab = [(R[0]['chan'][i], R[0]['chan'][j]) for i, j in zip(*iu)]
        rec = dict(agent=R[0]['agent'], animal=R[0]['animal'], target=tgt, M=M, n=len(R))
        for k, v in c.items():
            rec[k] = dict(C=held(v, conc), Cneg=held(v, lambda a, b: conc(a, b, True)), zres=v['zres'])
            if k == 'drug':
                rec[k]['e3_rho'], rec[k]['e3_gain_frac'] = e3(v)
                node = np.array([v['full'][(iu[0] == i) | (iu[1] == i)].mean() for i in range(M)])
                pn = np.clip(node, 0, None)
                rec[k]['top5_node_share'] = float(np.sort(pn)[-5:].sum() / pn.sum()) if pn.sum() > 0 else np.nan
                rec[k]['additive_r2'] = additive_r2(v['full'], M)
                top = np.argsort(v['full'])[-int(np.ceil(TOP * len(lab))):]
                dist = np.array([abs(a - b) for a, b in lab])
                rec[k]['top_label_dist_median'], rec[k]['all_label_dist_median'] = float(np.median(dist[top])), float(np.median(dist))
        rec['_maps'] = {k: dict(zip(lab, v['full'])) for k, v in c.items()}
        S[u] = rec
    out = dict(sessions={u: {k: v for k, v in r.items() if k != '_maps'} for u, r in S.items()})
    # E1 and E3 at the animal level, primary agents pooled
    A = {}
    for u, r in S.items():
        if r['agent'] in PRIMARY and 'drug' in r and 'eo' in r:
            A.setdefault(r['animal'], []).append(r)
    an = {a: dict(C=np.mean([r['drug']['C'] for r in L]), C_eo=np.mean([r['eo']['C'] for r in L]),
                  gain=np.mean([r['drug']['e3_gain_frac'] for r in L]), rho=np.mean([r['drug']['e3_rho'] for r in L]), n=len(L))
          for a, L in A.items()}
    beats = sum(v['C'] > v['C_eo'] for v in an.values()); med = float(np.median([v['C'] for v in an.values()]))
    e1 = ('CONCENTRATED' if beats == len(an) == 4 and med >= .05 else
          'DIFFUSE EXCESS' if beats == len(an) == 4 else 'NOT CONCENTRATED')
    g = [v['gain'] for v in an.values()]
    e3v = 'gain (a)' if all(x > .6 for x in g) else 'fail to gain (b)' if all(x < .4 for x in g) else 'mixed'
    # E2: within-animal cross-session reproducibility of ẽ maps on shared electrode pairs
    e2 = {}
    for a, L in A.items():
        us = [u for u, r in S.items() if r['animal'] == a and r['agent'] in PRIMARY and 'drug' in r]
        if len(us) < 2: continue
        def mean_rho(key):
            v = []
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    m1, m2 = S[us[i]]['_maps'][key], S[us[j]]['_maps'][key]; sh = sorted(set(m1) & set(m2))
                    v.append(spearmanr([m1[e] for e in sh], [m2[e] for e in sh])[0])
            return float(np.mean(v))
        def overlap(key):  # descriptive, added before outputs: Jaccard of top-1% edges between sessions
            v = []
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    m1, m2 = S[us[i]]['_maps'][key], S[us[j]]['_maps'][key]; sh = sorted(set(m1) & set(m2)); k = int(np.ceil(TOP * len(sh)))
                    t1 = set(sorted(sh, key=lambda e: m1[e])[-k:]); t2 = set(sorted(sh, key=lambda e: m2[e])[-k:])
                    v.append(len(t1 & t2) / len(t1 | t2))
            return float(np.mean(v))
        e2[a] = dict(drug=mean_rho('drug'), eo=mean_rho('eo'), n_sessions=len(us), top_jaccard_drug=overlap('drug'), top_jaccard_eo=overlap('eo'))
    e2v = 'REPRODUCIBLE' if e2 and all(v['drug'] - v['eo'] >= .10 for v in e2.values()) else 'NOT REPRODUCIBLE'
    # descriptive: other agents and sleep at the animal level
    desc = {}
    for ag in ('MD', 'SLP'):
        L = [r for r in S.values() if r['agent'] == ag and 'drug' in r and 'eo' in r]
        desc[ag] = dict(C=float(np.mean([r['drug']['C'] for r in L])) if L else None, C_eo=float(np.mean([r['eo']['C'] for r in L])) if L else None, n=len(L))
    zres = {ag: dict(mean=float(np.mean([r['drug']['zres'] for r in S.values() if r['agent'] == ag])),
                     down=int(sum(r['drug']['zres'] < 0 for r in S.values() if r['agent'] == ag)),
                     n=int(sum(r['agent'] == ag for r in S.values())))
            for ag in ('KT', 'KTMD', 'MD', 'PF', 'SLP') if any(r['agent'] == ag for r in S.values())}
    out.update(animals=an, E1=dict(verdict=e1, beats_eo=int(beats), median_C=med), E2=dict(verdict=e2v, per_animal=e2),
               E3=dict(verdict=e3v), descriptive=desc, zres_by_agent=zres)
    json.dump(out, open(sys.argv[2], 'w'), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('E1', 'E2', 'E3', 'animals', 'descriptive', 'zres_by_agent')}, indent=1, default=float))
