"""POST HOC checks on the edge arm (written after edges_score.py had been run; not part of the frozen rule).
1. Strength: does the excess follow pairs whose coupling strengthens? Top-1% edges by wake-g decile; Spearman of ẽ
   with drug-state g and with Δg; C and within-animal ρ after binning on wake g × drug g (10 × 10).
2. Surrogate averaging: z_res with the three-draw mean k_s against one draw's noise level (k_s + N(0, ksd²) per edge).
Usage: python edges_posthoc.py <dir with edges_*.npz> <out.json>
"""
import sys, json, numpy as np
from scipy.stats import spearmanr
from edges_score import load, std, conc, PRIMARY, TOP


def deciles(v): return np.minimum((np.argsort(np.argsort(v)) * 10) // len(v), 9)


def session(R, rng):
    st = np.array([r['state'] for r in R]); X = np.array([r['X'] for r in R]); e = X[:, 1] - X[:, 3]
    tgt = 'sleep' if R[0]['agent'] == 'SLP' else 'drug'; out = {}
    gw = X[st == 'awake_ec', 0].mean(0); dw = deciles(gw)
    for name, s in (('drug', tgt), ('eo', 'awake_eo')):
        if not (st == s).any(): continue
        gs = X[st == s, 0].mean(0); ds = deciles(gs); cell = dw * 10 + ds
        def adj(d, c):
            m = np.zeros(100); n = np.bincount(c, minlength=100); m[n > 0] = np.bincount(c, d, 100)[n > 0] / n[n > 0]; return d - m[c]
        idx = lambda s_, h: np.flatnonzero(st == s_)[h::2] if h is not None else np.flatnonzero(st == s_)
        dm = lambda h: e[idx(s, h)].mean(0) - e[idx('awake_ec', h)].mean(0)
        a1 = lambda d: d - np.array([d[dw == b].mean() for b in range(10)])[dw]  # frozen (wake-only) adjustment
        full1, full2 = a1(dm(None)), adj(dm(None), cell)
        h2 = [adj(dm(0), cell), adj(dm(1), cell)]
        top = np.argsort(full1)[-int(np.ceil(TOP * len(full1))):]
        noisy = []
        for x in X:  # one-draw noise added to the three-draw mean surrogate profile
            k1 = x[3] + rng.normal(0, 1, len(x[3])) * x[4]
            noisy.append(np.mean(std(x[0]) * std(x[1])) - np.mean(std(x[0]) * std(k1)))
        noisy = np.array(noisy); mean3 = np.array([np.mean(std(x[0]) * std(x[1])) - np.mean(std(x[0]) * std(x[3])) for x in X])
        out[name] = dict(
            top_wake_decile_hist=np.bincount(dw[top], minlength=10).tolist(),
            rho_e_gstate=float(spearmanr(full1, gs)[0]), rho_e_dg=float(spearmanr(full1, gs - gw)[0]),
            C_joint=float(np.mean([conc(h2[0], h2[1]), conc(h2[1], h2[0])])),
            zres_mean3=float(mean3[idx(s, None)].mean() - mean3[idx('awake_ec', None)].mean()),
            zres_onedraw=float(noisy[idx(s, None)].mean() - noisy[idx('awake_ec', None)].mean()),
            map=full2)
    return out


if __name__ == '__main__':
    rows = load(sys.argv[1]); rng = np.random.default_rng(20261117); S = {}
    for u in sorted({r['unit'] + '|' + r['agent'] for r in rows}):
        R = [r for r in rows if r['unit'] + '|' + r['agent'] == u]; M = len(R[0]['chan']); iu = np.triu_indices(M, 1)
        lab = [(R[0]['chan'][i], R[0]['chan'][j]) for i, j in zip(*iu)]
        c = session(R, rng); S[u] = dict(agent=R[0]['agent'], animal=R[0]['animal'], **{k: {kk: vv for kk, vv in v.items() if kk != 'map'} for k, v in c.items()},
                                        _maps={k: dict(zip(lab, v['map'])) for k, v in c.items()})
    an = {}
    for u, r in S.items():
        if r['agent'] in PRIMARY: an.setdefault(r['animal'], []).append(u)
    res = {}
    for a, us in an.items():
        def mrho(key):
            v = []
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    m1, m2 = S[us[i]]['_maps'][key], S[us[j]]['_maps'][key]; sh = sorted(set(m1) & set(m2))
                    v.append(spearmanr([m1[e] for e in sh], [m2[e] for e in sh])[0])
            return float(np.mean(v)) if v else np.nan
        res[a] = dict(C_joint=float(np.mean([S[u]['drug']['C_joint'] for u in us])), C_joint_eo=float(np.mean([S[u]['eo']['C_joint'] for u in us])),
                      rho_joint=mrho('drug'), rho_joint_eo=mrho('eo'),
                      rho_e_gstate=float(np.mean([S[u]['drug']['rho_e_gstate'] for u in us])), rho_e_dg=float(np.mean([S[u]['drug']['rho_e_dg'] for u in us])),
                      top_wake_decile_hist=np.sum([S[u]['drug']['top_wake_decile_hist'] for u in us], 0).tolist())
    z = {ag: dict(mean3=float(np.mean([r['drug']['zres_mean3'] for r in S.values() if r['agent'] == ag])),
                  onedraw=float(np.mean([r['drug']['zres_onedraw'] for r in S.values() if r['agent'] == ag])),
                  onedraw_down=int(sum(r['drug']['zres_onedraw'] < 0 for r in S.values() if r['agent'] == ag)),
                  eo_onedraw_sd=float(np.std([r['eo']['zres_onedraw'] for r in S.values() if 'eo' in r], ddof=1)),
                  n=int(sum(r['agent'] == ag for r in S.values())))
         for ag in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
    out = dict(animals=res, zres=z); json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
