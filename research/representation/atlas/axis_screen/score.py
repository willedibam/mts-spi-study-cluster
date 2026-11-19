"""Axis screen score (prereg.md): families F1 (beyond IAAFT), F2 (lag beyond zero-phase floor), F3 (residual beyond the
stationary null); session screen, axis gates, and the reported controls.

  python -m research.representation.atlas.axis_screen.score <screen-<arm>.npz> <manifest.json> <out.json> [stage-a-ids.txt]
"""
import sys, json
import numpy as np
from scipy.stats import ttest_1samp
from research.representation.atlas.axis_screen.stats import KEYS, PYSPI

KSG = 'mi_kraskov_NN-4_DCE-AUTO'
HIGHER = {'rho', 'rho2', 'tau', 'tau2', 'mcd', 'mcd2', 'pcorr_mcd', 'l1', 'linf', 'dcorr', 'pec', 'pec_log', 'ksg'}
LAG = {'icoh_lo', 'icoh_bb', 'dswpli', 'dswpli_lo', 'dswpli_hi', 'xcmax', 'xc2max'}
DRUGS, REF = ('KT', 'KTMD', 'MD', 'PF'), 'awake_ec'
V = dict(real=0, iaaft=(1, 2, 3), stationary=4, zero_phase=5, bias=6)
NULL = {'F1': V['iaaft'], 'F2': (V['zero_phase'],), 'F3': (V['stationary'],)}
COVS = {'F1': (0, 1, 2, 6), 'F2': (0, 1, 2, 3, 4, 5), 'F3': (0, 1, 2, 3, 4, 5)}
Q, FIN, REL, CEIL = .05, .95, .3, .98


def bh(p):
    p = np.where(np.isfinite(p), p, 1.)
    n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def name(k):
    return KSG if k == 'ksg' else PYSPI[k]


def load(path, manifest, stage_a=None):
    D = np.load(path, allow_pickle=False)
    rows = {r['id']: r for r in json.load(open(manifest))['rows']}
    ids = [str(i) for i in D['ids']]
    R = [rows[i] for i in ids]
    keys = [str(k) for k in D['keys']]
    g_index = {str(g): k for k, g in enumerate(D['groups'])}
    gi = np.array([g_index[str(g)] for g in D['group']])
    info = dict(unit=np.array([r['unit'] for r in R]), state=np.array([r['state'] for r in R]),
                agent=np.array([r.get('agent', '') for r in R]), animal=np.array([r.get('animal', '') for r in R]),
                start=np.array([r.get('start_s', r.get('onset_s', np.nan)) for r in R], float), sub=np.array([str(s) for s in D['sub']]),
                stage=np.array(['A' if stage_a and i in stage_a else 'B' for i in ids]))
    return D, keys, gi, info


def coordinates(keys):
    """(label, space, i, j, a, b): all-channel pairs, then KSG (32 channels) against every other statistic."""
    C = []
    S = len(keys)
    for i in range(S):
        for j in range(i + 1, S):
            C.append((f'{keys[i]}|{keys[j]}', 'all', i, j, keys[i], keys[j]))
    for i in range(S):
        C.append((f'{keys[i]}|ksg', 'sub32', i, S, keys[i], 'ksg'))
    return C


def family(a, b):
    out = []
    if a in HIGHER or b in HIGHER:
        out.append('F1')
    else:
        if a in LAG or b in LAG:
            out.append('F2')
        out.append('F3')
    return out


def score(path, manifest, output, stage_a_path=None):
    from research.representation.atlas.assumptions import classify, READINGS
    stage_a = set(open(stage_a_path).read().split()) if stage_a_path else None
    D, keys, gi, I = load(path, manifest, stage_a)
    Zall, Z32, Zdc, cov, rel, zmean = D['Z'], D['Z32'], D['Zdc'], D['cov'], D['rel'], D['zmean']
    groups = [str(g) for g in D['groups']]
    S = len(keys)
    coords = coordinates(keys)
    unit, state, agent, animal = I['unit'], I['state'], I['agent'], I['animal']
    ag = {u: a for u, a in zip(unit, agent)}; an = {u: a for u, a in zip(unit, animal)}
    sessions = sorted(u for u in set(unit) if ag[u] in DRUGS and ((unit == u) & (state == 'drug')).any() and ((unit == u) & (state == REF)).any())
    animals = sorted({an[u] for u in sessions})
    sa = np.array([an[u] for u in sessions])
    sleep_units = sorted(u for u in set(unit) if ((unit == u) & (state == 'sleep')).any() and ((unit == u) & (state == REF)).any())
    G = lambda u, s: np.flatnonzero((unit == u) & (state == s))
    relw = rel[gi]  # (n, 7, S+1)
    mi_g = keys.index('mi_g')

    def zser(c, versions, dc=False, disatt=False, Zs=None):
        """Per-window mean over the given versions of z (optionally disattenuated) for coordinate c."""
        _, space, i, j, _, _ = c
        Z = (Zdc if dc else Zall) if space == 'all' else Z32
        if Zs is not None:
            Z = Zs
        x = Z[:, list(versions), i, j].astype(float)
        if disatt:
            ri = relw[:, list(versions), i if space == 'all' or i < S else S]
            rj = relw[:, list(versions), j if j < S else S]
            with np.errstate(all='ignore'):
                x = x / np.sqrt(np.clip(ri, 1e-3, None) * np.clip(rj, 1e-3, None))
        return np.nanmean(x, 1)

    def net(c, fam, **kw):
        return zser(c, (V['real'],), **kw) - zser(c, NULL[fam], **kw)

    def sdelta(x, units, s, ref=REF, sub=None):
        out = []
        for u in units:
            d = G(u, s) if sub is None else np.flatnonzero((unit == u) & (I['sub'] == sub))
            r = G(u, ref)
            out.append(np.nanmean(x[d]) - np.nanmean(x[r]) if len(d) and len(r) else np.nan)
        return np.array(out)

    def split_half(x):
        out = []
        for u in sessions:
            r = G(u, REF); r = r[np.argsort(I['start'][r])]
            h = len(r) // 2
            out.append(np.nanmean(x[r[-h:]]) - np.nanmean(x[r[:h]]))
        return np.array(out)

    def animal_means(d, units=None):
        units = sessions if units is None else units
        a_ = np.array([an[u] for u in units])
        return np.array([np.nanmean(d[a_ == k]) for k in animals])

    def adjusted(x, fam):
        out = []
        for u in sessions:
            ix = np.concatenate([G(u, REF), G(u, 'drug')])
            y = x[ix]; d = (state[ix] == 'drug').astype(float)
            C = cov[ix][:, list(COVS[fam])].astype(float)
            C = C[:, np.isfinite(C).all(0) & (np.nanstd(C, 0) > 1e-12)]
            C = (C - C.mean(0)) / C.std(0)
            X = np.column_stack([np.ones(len(ix)), d, C])
            ok = np.isfinite(y)
            beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            out.append(beta[1])
        return np.array(out)

    # usability (blind to contrast)
    used_states = (REF, 'awake_eo', 'drug', 'recovery_ec')
    mask = np.isin(unit, sessions) & np.isin(state, used_states)
    rel_ref = {}
    for k in range(S + 1):
        rel_ref[k] = float(np.nanmean([rel[groups.index(f'{u}__{REF}'), 0, k] for u in sessions]))

    MI = ('mi_g|ksg', 'sub32', mi_g, S, 'mi_g', 'ksg')
    mi_net = net(MI, 'F1')

    def row(c, fam, p=None, q=None, full=True):
        lab, space, i, j, a, b = c
        x = net(c, fam)
        d = sdelta(x, sessions, 'drug')
        am = animal_means(d)
        sh = split_half(x)
        null_sd = np.nanstd(sh, ddof=1) / np.sqrt(2)
        eo = sdelta(x, sessions, 'awake_eo')
        r = dict(pair=lab, family=fam, spis=[name(a), name(b)], space=space, cls=classify(name(a), name(b)),
                 mean=float(np.nanmean(d)), sessions_same_sign=int(np.sum(np.sign(d) == np.sign(np.nanmean(d)))), n=len(sessions),
                 p=p, q=q, animals=dict(zip(animals, am.round(5).tolist())), t3=float(ttest_1samp(am, 0).statistic),
                 p3=float(ttest_1samp(am, 0).pvalue), null_sd=float(null_sd), sd_multiple=float(np.nanmean(d) / null_sd),
                 split_half=float(np.nanmean(sh)), eyes_open=float(np.nanmean(eo)),
                 z0=float(np.nanmean([np.nanmean(zser(c, (0,))[G(u, REF)]) for u in sessions])),
                 reliability=[rel_ref[i], rel_ref[j if j < S else S]])
        rd = READINGS.get(r['cls'], (None, None, None))
        r.update(character=rd[0], literal=rd[1], confound=rd[2])
        if not full:
            return r
        adj = adjusted(x, fam); xd = net(c, fam, disatt=True); dd = sdelta(xd, sessions, 'drug')
        r.update(adjusted=float(np.nanmean(adj)), adjusted_animals=animal_means(adj).round(5).tolist(),
                 disattenuated=float(np.nanmean(dd)), disattenuated_animals=animal_means(dd).round(5).tolist())
        m = r['mean']
        gates = {
            'animal_t3': bool(r['p3'] < .05 and len(set(np.sign(am))) == 1),
            'size_3sd': bool(abs(m) >= 3 * null_sd),
            'split_half': bool(abs(r['split_half']) <= .25 * abs(m)),
            'eyes_open': bool(abs(r['eyes_open']) <= .33 * abs(m)),
            'covariates': bool(np.sign(r['adjusted']) == np.sign(m) and abs(r['adjusted']) >= .5 * abs(m)
                               and all(np.sign(r['adjusted_animals']) == np.sign(m))),
            'reliability': bool(all(np.sign(r['disattenuated_animals']) == np.sign(m))),
        }
        r['gates'] = gates; r['axis'] = bool(all(gates.values()))
        real = zser(c, (0,)); nul = zser(c, NULL[fam])
        r['real_delta'] = float(np.nanmean(sdelta(real, sessions, 'drug')))
        r['null_delta'] = float(np.nanmean(sdelta(nul, sessions, 'drug')))
        if fam == 'F3':
            r['bias'] = float(np.nanmean(sdelta(zser(c, (V['bias'],)) - zser(c, (V['stationary'],)), sessions, 'drug')))
        if fam == 'F2':
            r['mean_lag'] = float(np.nanmean(sdelta(zser(c, (V['stationary'],)) - zser(c, (V['zero_phase'],)), sessions, 'drug')))
            r['residual'] = float(np.nanmean(sdelta(real - zser(c, (V['stationary'],)), sessions, 'drug')))
        r['per_agent'] = {k: float(np.nanmean([dv for u, dv in zip(sessions, d) if ag[u] == k])) for k in DRUGS}
        sl = sdelta(x, sleep_units, 'sleep'); deep = sdelta(x, sleep_units, None, sub='sleep_deep')
        rec = sdelta(x, sessions, 'recovery_ec')
        r.update(sleep=float(np.nanmean(sl)), sleep_ratio=float(np.nanmean(sl) / m), sleep_deep=float(np.nanmean(deep)),
                 sleep_units_same_sign=f"{int(np.sum(np.sign(sl) == np.sign(m)))}/{int(np.isfinite(sl).sum())}",
                 recovery=float(np.nanmean(rec)), recovery_ratio=float(np.nanmean(rec) / m))
        if space == 'all':
            r['dc'] = float(np.nanmean(sdelta(net(c, fam, dc=True), sessions, 'drug')))
            r['node_r2'] = [float(np.nanmean(D['R2'][np.isin(unit, sessions) & (state == REF), 0, k])) for k in (i, j)]
            zm = lambda u, s, ver: float(np.mean([zmean[groups.index(f'{u}__{s}'), v, i, j] for v in ver]))
            mp = [zm(u, 'drug', (0,)) - zm(u, REF, (0,)) - (zm(u, 'drug', NULL[fam]) - zm(u, REF, NULL[fam])) for u in sessions]
            r['mean_profile_delta'] = float(np.mean(mp))
        r['reliability_change'] = [float(np.nanmean([rel[groups.index(f'{u}__drug'), 0, k] - rel[groups.index(f'{u}__{REF}'), 0, k]
                                                     for u in sessions])) for k in (i, j if j < S else S)]
        st = I['stage']
        r['stages'] = {s_: float(np.nanmean([dv for u, dv in zip(sessions, d) if st[unit == u][0] == s_])) for s_ in ('A', 'B')}
        dm = np.isin(unit, sessions) & (state == 'drug')
        xs, ms = x.copy(), mi_net.copy()
        for u in sessions:
            k = G(u, 'drug'); xs[k] -= np.nanmean(xs[k]); ms[k] -= np.nanmean(ms[k])
        ok = dm & np.isfinite(xs) & np.isfinite(ms)
        r['r_with_mi_net'] = float(np.corrcoef(xs[ok], ms[ok])[0, 1]) if lab != 'mi_g|ksg' else 1.
        if r['cls'] is None:
            r['links'] = links(c, fam)
        return r

    def links(c, fam):
        _, space, i, j, a, b = c
        if space != 'all':
            return []
        out = []
        for k, m_ in enumerate(keys):
            if k in (i, j):
                continue
            c1, c2 = classify(name(a), name(m_)), classify(name(m_), name(b))
            if c1 is None or c2 is None:
                continue
            Zp = Zall.astype(float)
            ab, am_, mb = Zp[:, :, i, j], Zp[:, :, i, k], Zp[:, :, k, j]
            with np.errstate(all='ignore'):
                part = (ab - am_ * mb) / np.sqrt((1 - am_ ** 2) * (1 - mb ** 2))
            pn = part[:, 0] - np.nanmean(part[:, list(NULL[fam])], 1)
            l1 = Zp[:, 0, i, k] - np.nanmean(Zp[:, list(NULL[fam]), i, k], 1)
            l2 = Zp[:, 0, k, j] - np.nanmean(Zp[:, list(NULL[fam]), k, j], 1)
            out.append(dict(via=m_, link1=[f'{a}|{m_}', c1, float(np.nanmean(sdelta(l1, sessions, 'drug')))],
                            link2=[f'{m_}|{b}', c2, float(np.nanmean(sdelta(l2, sessions, 'drug')))],
                            partial=float(np.nanmean(sdelta(pn, sessions, 'drug')))))
        return out

    res = dict(sessions=sessions, animals=animals, sleep_units=sleep_units, keys=keys, families={})
    for fam in ('F1', 'F2', 'F3'):
        C = [c for c in coords if fam in family(c[4], c[5])]
        fin = np.ones(len(C), bool); z0 = np.zeros(len(C)); relok = np.zeros(len(C), bool)
        P, Dm, every = np.full(len(C), np.nan), np.full(len(C), np.nan), np.zeros(len(C), bool)
        for n_, c in enumerate(C):
            for ver in ((0,),) + tuple((v,) for v in NULL[fam]):
                z = zser(c, ver)
                for u in sessions:
                    for s in used_states:
                        ix = G(u, s)
                        if len(ix) and np.isfinite(z[ix]).mean() < FIN:
                            fin[n_] = False
            z0[n_] = np.nanmean([np.nanmean(zser(c, (0,))[G(u, REF)]) for u in sessions])
            _, space, i, j, _, _ = c
            relok[n_] = rel_ref[i] >= REL and rel_ref[j if j < S else S] >= REL
            d = sdelta(net(c, fam), sessions, 'drug')
            P[n_] = ttest_1samp(d, 0, nan_policy='omit').pvalue; Dm[n_] = np.nanmean(d)
            every[n_] = len(set(np.sign(animal_means(d)))) == 1
        base = fin & (np.abs(z0) < CEIL)
        usable = base & relok
        q = np.full(len(C), np.nan); q[usable] = bh(P[usable])
        scr = usable & (q <= Q) & every
        qn = np.full(len(C), np.nan); qn[base] = bh(P[base])
        scr_norel = base & ~relok & (qn <= Q) & every
        rows = sorted([row(C[k], fam, float(P[k]), float(q[k])) for k in np.flatnonzero(scr)], key=lambda r: r['p'])
        res['families'][fam] = dict(coordinates=len(C), finite=int(fin.sum()), usable=int(usable.sum()), screened=int(scr.sum()),
                                    axes=[r['pair'] for r in rows if r['axis']], rows=rows,
                                    screened_no_reliability=sorted([row(C[k], fam, float(P[k]), float(qn[k]), full=False)
                                                                    for k in np.flatnonzero(scr_norel)], key=lambda r: r['p']),
                                    top_not_screened=[dict(pair=C[k][0], mean=float(Dm[k]), p=float(P[k]), q=float(q[k]), every=bool(every[k]))
                                                      for k in np.argsort(np.where(usable & ~scr, P, np.inf))[:15]])
        print(f"{fam}: coords {len(C)} finite {fin.sum()} usable {usable.sum()} screened {scr.sum()} axes {len(res['families'][fam]['axes'])}", flush=True)
    res['anchor'] = row(MI, 'F1')
    named = ['linf|l2', 'linf|r2', 'l1|l2', 'r|mcd', 'r2|mcd2', 'pcorr|pcorr_mcd', 'r|rho', 'rho|tau', 'r2|dcorr', 'pec|pec_log',
             'pec|coh_bb', 'r2|xc2max', 'r2|coh_lo', 'coh_lo|coh_hi', 'ppc|dswpli', 'ppc_lo|dswpli_lo', 'coh_lo|icoh_lo', 'plv_hi|dswpli_hi',
             'r2|pcorr', 'coh_bb|plv', 'linf|ksg', 'linf|mi_g']
    res['named'] = []
    for lab in named:
        c = [x for x in coords if x[0] == lab or x[0] == '|'.join(lab.split('|')[::-1])]
        if c:
            for fam in family(c[0][4], c[0][5]):
                res['named'].append(row(c[0], fam))
    json.dump(res, open(output, 'w'), indent=1, default=float)
    print('anchor', {k: res['anchor'][k] for k in ('mean', 't3', 'sd_multiple', 'split_half', 'eyes_open', 'axis')}, res['anchor']['gates'])


if __name__ == '__main__':
    score(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
