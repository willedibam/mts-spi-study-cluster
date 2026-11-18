"""Phase 3 scorer for prereg.md, Revision 1 (drop test decides broad against localised; shares descriptive).
Usage: python movers_score.py <movers.json> <edges dir> <out.json>"""
import sys, json, glob, numpy as np
from pathlib import Path

PRIMARY, DRAWS, CONF = ('KT', 'KTMD', 'PF'), 20, .02


def verdict(f, rand):
    lo, hi = float(np.min(rand)), float(np.max(rand))
    if f < .8 and f < lo: return 'LOCALISED'
    if f >= .9 or lo <= f <= hi: return 'BROAD'
    return 'PARTIAL'


def node(q, M):
    iu = np.triu_indices(M, 1); c = np.zeros(M); np.add.at(c, iu[0], q / 2); np.add.at(c, iu[1], q / 2); return c


if __name__ == '__main__':
    m = json.loads(Path(sys.argv[1]).read_text()); spis = m['spis']; K = len(spis); a, b = np.triu_indices(K, 1)
    U = {Path(f).stem: dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(f'{sys.argv[2]}/*.npz'))}
    coords = [int(c) for c in U[next(iter(U))]['coords']]; info = {mv['coord']: mv for mv in m['movers']}
    prim = [u for u, d in U.items() if str(d['agent']) in PRIMARY and 'drug_all' in d]
    feats = []
    for n, c in enumerate(coords):
        mv = info.get(c); sg = np.sign(m['net_h0_by_coord'][str(c)])
        D = np.array([U[u][f'drop_{h}'][n] for u in prim for h in (0, 1)])  # (sessions x halves, 3 + 2 DRAWS)
        full = D[:, 0].sum(); f_hot, f_node = D[:, 1].sum() / full, D[:, 2].sum() / full
        f_rand = D[:, 3:3 + DRAWS].sum(0) / full; f_rnode = D[:, 3 + DRAWS:].sum(0) / full
        heldout = full / len(D)
        tiered = abs(heldout) >= CONF
        animals = {}
        for u in prim:
            an = str(U[u]['animal']); x = animals.setdefault(an, [0, 0, 0])
            for h in (0, 1):
                x[0] += U[u][f'drop_{h}'][n, 0]; x[1] += U[u][f'drop_{h}'][n, 1]; x[2] += U[u][f'drop_{h}'][n, 3:3 + DRAWS].mean()
        # carriers by electrode-label pair, and nodes by electrode label
        tops, eos, nodes = {}, {}, {}
        for u in prim:
            d = U[u]; ch = [int(x) for x in d['chan']]; M = len(ch); iu = np.triu_indices(M, 1)
            lab = [(ch[i], ch[j]) for i, j in zip(*iu)]; k = int(np.ceil(.01 * len(lab)))
            tops[u] = {lab[e] for e in np.argsort(-sg * d['drug_all'][n])[:k]}
            eos[u] = {lab[e] for e in np.argsort(-sg * d['awake_eo_all'][n])[:k]}
            for l, x in zip(ch, sg * node(d['drug_all'][n], M)): nodes.setdefault(str(d['animal']), {}).setdefault(l, []).append(x)
        jac = {'same agent': [], 'different agent': [], 'eyes-open maps': []}
        for i, u in enumerate(prim):
            for v in prim[i + 1:]:
                if str(U[u]['animal']) != str(U[v]['animal']): continue
                J = lambda p, q: len(p & q) / len(p | q) if p | q else np.nan
                jac['same agent' if str(U[u]['agent']) == str(U[v]['agent']) else 'different agent'].append(J(tops[u], tops[v]))
                jac['eyes-open maps'].append(J(eos[u], eos[v]))
        topnodes = {an: sorted(L, key=lambda l: -np.mean(L[l]))[:6] for an, L in nodes.items()}
        feats.append(dict(coord=c, name=f'{spis[a[c]]}|{spis[b[c]]}', a=spis[a[c]], b=spis[b[c]], cls=mv['cls'] if mv else None,
                          chain=mv['chain'] if mv else None, modules=mv['modules'] if mv else None, anchor=(c == m['anchor']),
                          z0=mv['z0'] if mv else None, net_h1=mv['net_h1'] if mv else None, h1_by_agent=mv['h1_by_agent'] if mv else None,
                          heldout_dz=float(heldout), f_hot=float(f_hot), f_rand=[float(f_rand.min()), float(f_rand.mean()), float(f_rand.max())],
                          f_node=float(f_node), f_rnode=[float(f_rnode.min()), float(f_rnode.mean()), float(f_rnode.max())],
                          verdict_edges=verdict(f_hot, f_rand) if tiered else 'NOT TIERED', verdict_nodes=verdict(f_node, f_rnode) if tiered else 'NOT TIERED',
                          by_animal={k: dict(f_hot=v[1] / v[0], f_rand=v[2] / v[0]) for k, v in animals.items()},
                          jaccard={k: [float(np.nanmean(v)) if v else None, len(v)] for k, v in jac.items()}, top_nodes=topnodes))
    C = np.nanmean([np.atleast_2d(np.corrcoef(np.array([U[u]['drug_all'][n] for n in range(len(coords))]))) for u in prim], 0)
    count = lambda key: {t: sum(f[key] == t for f in feats) for t in ('BROAD', 'PARTIAL', 'LOCALISED', 'NOT TIERED')}
    out = dict(phase1={k: m[k] for k in ('usable', 'selected', 'confirmed', 'eo_selected', 'eo_confirmed', 'fdp_estimate', 'confirmed_positive', 'by_class', 'per_spi')},
               verdicts_edges=count('verdict_edges'), verdicts_nodes=count('verdict_nodes'), substrate_corr=C.tolist(),
               substrate_mean_offdiag=float(np.nanmean(C[~np.eye(len(C), dtype=bool)])), features=feats)
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('verdicts_edges', 'verdicts_nodes', 'substrate_mean_offdiag')}, indent=1))
    for f in feats:
        print(f"{f['name'][:58]:58s} {str(f['cls'])[:30]:30s} dz {f['heldout_dz']:+.3f} f_hot {f['f_hot']:.2f} [{f['f_rand'][0]:.2f},{f['f_rand'][2]:.2f}] {f['verdict_edges']:10s} node {f['f_node']:.2f} {f['verdict_nodes']}")
