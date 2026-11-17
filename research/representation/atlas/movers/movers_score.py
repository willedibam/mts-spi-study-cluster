"""Phase 3 scorer for prereg.md (frozen with it). Usage: python movers_score.py <movers.json> <edges dir> <out.json>"""
import sys, json, glob, numpy as np
from pathlib import Path

PRIMARY = ('KT', 'KTMD', 'PF')
TIER_E, TIER_N = (.02, .05), (.10, .25)


def tier(x, t): return 'BROAD' if x < t[0] else 'ENRICHED' if x < t[1] else 'CONCENTRATED'


def held_share(sel, meas, k, sg):
    top = np.argsort(-sg * sel)[:k]; tot = meas.sum()
    return float(meas[top].sum() / tot) if abs(tot) > 1e-12 else np.nan


def node(q, M):
    iu = np.triu_indices(M, 1); c = np.zeros(M); np.add.at(c, iu[0], q / 2); np.add.at(c, iu[1], q / 2); return c


def e50(v, sg):
    s = np.sort(sg * v)[::-1]; tot = s.sum()
    return float(np.argmax(np.cumsum(s) >= .5 * tot) + 1) / len(s) if tot > 0 else np.nan


if __name__ == '__main__':
    m = json.loads(Path(sys.argv[1]).read_text()); spis = m['spis']; K = len(spis); a, b = np.triu_indices(K, 1)
    U = {Path(f).stem: dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(f'{sys.argv[2]}/*.npz'))}
    coords = list(U[next(iter(U))]['coords']); info = {mv['coord']: mv for mv in m['movers']}
    prim = [u for u, d in U.items() if str(d['agent']) in PRIMARY and 'drug_all' in d]
    feats = []
    for n, c in enumerate(coords):
        mv = info.get(c); name = f'{spis[a[c]]}|{spis[b[c]]}'
        sg = np.sign(mv['net_h0']) if mv else np.sign(np.sum([U[u]['drug_0'][n].sum() for u in prim]))
        S = {}
        for u in prim:
            d = U[u]; M = len(d['chan']); ke = max(1, int(np.ceil(.01 * len(d['drug_all'][n])))); kn = max(1, int(round(.05 * M)))
            q0, q1, qa = d['drug_0'][n], d['drug_1'][n], d['drug_all'][n]
            se = np.nanmean([held_share(q0, q1, ke, sg), held_share(q1, q0, ke, sg)])
            n0, n1 = node(q0, M), node(q1, M)
            sn = np.nanmean([held_share(n0, n1, kn, sg), held_share(n1, n0, kn, sg)])
            pr = float(np.abs(qa).sum() ** 2 / (len(qa) * np.sum(qa ** 2)))
            S[u] = dict(S_e=float(se), S_n=float(sn), E50=e50(qa, sg), PR=pr, dz=float(qa.sum()),
                        top_edges=set(np.argsort(-sg * qa)[:ke].tolist()), top_eo=set(np.argsort(-sg * d['awake_eo_all'][n])[:ke].tolist()),
                        nodes=node(qa, M), chan=[int(x) for x in d['chan']], agent=str(d['agent']), animal=str(d['animal']))
        med = lambda k: float(np.nanmedian([v[k] for v in S.values()]))
        # carrier reproducibility within animal (edge index sets are comparable only when channel lists match)
        jac = {'same agent': [], 'different agent': [], 'eo': []}
        L = list(S)
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                x, y = S[L[i]], S[L[j]]
                if x['animal'] != y['animal'] or x['chan'] != y['chan']: continue
                J = lambda p, q: len(p & q) / len(p | q)
                jac['same agent' if x['agent'] == y['agent'] else 'different agent'].append(J(x['top_edges'], y['top_edges']))
                jac['eo'].append(J(x['top_eo'], y['top_eo']))
        # recurring top nodes per animal (mean node contribution over that animal's sessions, by electrode label)
        nodes = {}
        for an in sorted({v['animal'] for v in S.values()}):
            acc = {}
            for v in S.values():
                if v['animal'] != an: continue
                for lab, x in zip(v['chan'], sg * v['nodes']): acc.setdefault(lab, []).append(x)
            mean = {k: np.mean(x) for k, x in acc.items()}; tot = sum(mean.values())
            top = sorted(mean, key=lambda k: -mean[k])[:6]
            nodes[an] = dict(top=top, share=float(sum(mean[k] for k in top) / tot) if tot else None)
        feats.append(dict(coord=int(c), name=name, cls=mv['cls'] if mv else None, chain=mv['chain'] if mv else None,
                          modules=mv['modules'] if mv else None, anchor=(c == m['anchor']), net_h1=mv['net_h1'] if mv else None,
                          h1_by_agent=mv['h1_by_agent'] if mv else None,
                          S_e=med('S_e'), S_n=med('S_n'), E50=med('E50'), PR=med('PR'), dz_median=med('dz'),
                          tier_edges=tier(med('S_e'), TIER_E), tier_nodes=tier(med('S_n'), TIER_N),
                          jaccard={k: (float(np.mean(v)) if v else None, len(v)) for k, v in jac.items()}, nodes=nodes,
                          maps={u: v['nodes'].tolist() for u, v in S.items()}))
    # shared substrate: per-session correlation between features' edge maps, averaged
    C = []
    for u in prim:
        X = np.array([U[u]['drug_all'][n] for n in range(len(coords))])
        C.append(np.atleast_2d(np.corrcoef(X)))
    C = np.nanmean(C, 0)
    tiers = {t: sum(f['tier_edges'] == t for f in feats) for t in ('BROAD', 'ENRICHED', 'CONCENTRATED')}
    ntiers = {t: sum(f['tier_nodes'] == t for f in feats) for t in ('BROAD', 'ENRICHED', 'CONCENTRATED')}
    out = dict(phase1={k: m[k] for k in ('usable', 'selected', 'confirmed', 'eo_selected', 'eo_confirmed', 'fdp_estimate', 'confirmed_positive')},
               tiers_edges=tiers, tiers_nodes=ntiers, substrate_corr=C.tolist(),
               substrate_mean_offdiag=float(np.nanmean(C[~np.eye(len(C), dtype=bool)])),
               features=[{k: v for k, v in f.items() if k != 'maps'} for f in feats])
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('phase1', 'tiers_edges', 'tiers_nodes', 'substrate_mean_offdiag')}, indent=1))
    for f in out['features'][:15]:
        print(f"{f['name'][:60]:60s} {str(f['cls'])[:28]:28s} h1 {f['net_h1'] if f['net_h1'] is None else round(f['net_h1'], 3)} S_e {f['S_e']:.3f} {f['tier_edges']:12s} S_n {f['S_n']:.3f} {f['tier_nodes']}")
