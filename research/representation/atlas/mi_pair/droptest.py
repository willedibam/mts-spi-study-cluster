"""POST HOC drop test on the MI pair (10-s, all channels): is the z fall carried by a few edges or nodes?
Select the edges (or nodes) carrying the change on one half; on the other half recompute net Δz = Δ[corr(g, k) - corr(g_s, k_s)]
with them removed (profiles re-standardised by the correlation itself); compare with random drops matched on wake leverage.
Retained fraction f = Σ_sessions Δz_drop / Σ_sessions Δz_full. Usage: python droptest.py <dir with edges_*.npz> <out.json> [draws]
"""
import sys, json, numpy as np
from edges_score import load

PRIMARY = ('KT', 'KTMD', 'PF')
zc = lambda x, y, keep: np.corrcoef(x[keep], y[keep])[0, 1]
std = lambda v: (v - v.mean()) / v.std()


def zres(X, keep):  # X: (5, E) g, k, gs, ks, ksd
    return zc(X[0], X[1], keep) - zc(X[2], X[3], keep)


def run(rows, frac, draws, rng):
    units = sorted({r['unit'] + '|' + r['agent'] for r in rows}); res = {}
    for u in units:
        R = [r for r in rows if r['unit'] + '|' + r['agent'] == u]
        if R[0]['agent'] not in PRIMARY: continue
        st = np.array([r['state'] for r in R]); X = np.array([r['X'] for r in R]); E = X.shape[2]
        M = len(R[0]['chan']); iu = np.triu_indices(M, 1)
        idx = lambda s, h: np.flatnonzero(st == s)[h::2]
        q = np.array([std(x[0]) * std(x[1]) - std(x[2]) * std(x[3]) for x in X]) / E  # per-edge contribution to z_res
        lev = np.abs(np.array([std(x[0]) * std(x[1]) for x in X[st == 'awake_ec']])).mean(0)
        bins = np.minimum((np.argsort(np.argsort(lev)) * 10) // E, 9)
        out = []
        for hs, hm in ((0, 1), (1, 0)):
            dq = q[idx('drug', hs)].mean(0) - q[idx('awake_ec', hs)].mean(0)
            sg = np.sign(dq.sum()); k = int(np.ceil(frac * E))
            hot = np.argsort(sg * dq)[::-1][:k]
            node = np.zeros(M); np.add.at(node, iu[0], dq / 2); np.add.at(node, iu[1], dq / 2)
            hn = np.argsort(sg * node)[::-1][:max(1, round(.05 * M))]
            hot_n = np.flatnonzero(np.isin(iu[0], hn) | np.isin(iu[1], hn))
            def dz(drop):
                keep = np.ones(E, bool); keep[drop] = False
                f = lambda s: np.mean([zres(X[i], keep) for i in idx(s, hm)])
                return f('drug') - f('awake_ec')
            full = dz(np.array([], int))
            rand = []
            for _ in range(draws):
                pick = np.concatenate([rng.choice(np.flatnonzero(bins == b), np.sum(bins[hot] == b), replace=False) for b in range(10)])
                rand.append(dz(pick))
            rnode = [dz(np.flatnonzero(np.isin(iu[0], s) | np.isin(iu[1], s)))
                     for s in (rng.choice(M, len(hn), replace=False) for _ in range(draws))]
            out.append(dict(full=full, hot=dz(hot), rand=rand, hot_node=dz(hot_n), rand_node=rnode, n_node_edges=len(hot_n)))
        res[u] = dict(agent=R[0]['agent'], animal=R[0]['animal'], halves=out)
    F = lambda key: [h[key] for v in res.values() for h in v['halves']]
    full = np.sum(F('full'))
    f_hot = np.sum(F('hot')) / full
    f_rand = np.sum(np.array(F('rand')), 0) / full  # per draw, pooled
    f_hn = np.sum(F('hot_node')) / full; f_rn = np.sum(np.array(F('rand_node')), 0) / full
    by_animal = {}
    for v in res.values():
        a = by_animal.setdefault(v['animal'], dict(full=0, hot=0, rand=0))
        for h in v['halves']: a['full'] += h['full']; a['hot'] += h['hot']; a['rand'] += np.mean(h['rand'])
    return dict(frac=frac, sessions=len(res), f_hot=float(f_hot), f_rand_mean=float(f_rand.mean()), f_rand_range=[float(f_rand.min()), float(f_rand.max())],
                f_hot_node=float(f_hn), f_rand_node_mean=float(f_rn.mean()), f_rand_node_range=[float(f_rn.min()), float(f_rn.max())],
                node_edges_dropped_median=float(np.median(F('n_node_edges'))),
                by_animal={k: dict(f_hot=v['hot'] / v['full'], f_rand=v['rand'] / v['full']) for k, v in by_animal.items()})


if __name__ == '__main__':
    rows = load(sys.argv[1]); draws = int(sys.argv[3]) if len(sys.argv) > 3 else 20; rng = np.random.default_rng(20261117)
    out = {str(f): run(rows, f, draws, rng) for f in (.01, .05)}
    json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
