"""gate/imbal/switch real and per-draw IAAFT for every window, surrogates identical to the z runs (amplitude-prereg.md).
    python gate_run.py mouse OUT.npz      # claar_sel (awake, iso) + claar_ind_sel (induction); ns = CAR + normal-scored, 3 draws
    python gate_run.py macaque OUT.npz    # zns_[AB]_*.npz rows from raw/nt10, raw/nt10b; 32-ch S(M), normal-scored, 1 draw
Out F[w] = [real, surr_1..] × [gate, imbal, switch]."""
import sys, json, glob, os, zlib, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from surr_pairs import multivariate_iaaft
from zgk_ns import ns
from gate import block_feats, top_edges
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1')
from research.representation.atlas.prepare import reference


def feats(v, E, seeds):
    return np.array([block_feats(v, E)] + [block_feats(multivariate_iaaft(v, np.random.default_rng(s), final='spectrum')[0], E) for s in seeds])


if __name__ == '__main__':
    kind, out = sys.argv[1], sys.argv[2]; jobs, rows = [], []
    if kind == 'mouse':
        prep = lambda w: ns(reference(w.astype(np.float64), 'car'))
        for src in ('claar_sel', 'claar_ind_sel'):
            for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
                s = os.path.basename(fn)[:-4]; R = json.loads(str(np.load(fn)['rows']))
                for i, r in enumerate(R): rows.append(dict(r, session=s, src=src, idx=i)); jobs.append((fn, i, [zlib.crc32(f'{s}:{i}'.encode()) * 10 + d for d in range(3)]))
        edges = {}
        for fn in sorted(glob.glob('claar_sel/sub-*.npz')):
            Z = np.load(fn); R = json.loads(str(Z['rows'])); edges[os.path.basename(fn)[:-4]] = top_edges([prep(Z['W'][i]) for i, r in enumerate(R) if r['state'] == 'awake'])
        load = lambda fn, i: prep(np.load(fn)['W'][i]); key = lambda r: r['session']
    else:
        S = lambda m: np.round(np.arange(32) * (m - 1) / 31).astype(int); src = {'A': 'raw/nt10', 'B': 'raw/nt10b'}
        man = {a: {r['id']: r for r in json.load(open(f'{d}/manifest.json'))['rows']} for a, d in src.items()}
        for fz in sorted(glob.glob('zns_[AB]_*.npz')):
            a = os.path.basename(fz)[4]; Z = np.load(fz); m0 = man[a][str(Z['id'][0])]['M']   # zns runs used rows[0]['M'] (zns_B_SLP: per row)
            for k, wid in enumerate(Z['id']):
                r = man[a][str(wid)]; m = r['M'] if os.path.basename(fz) == 'zns_B_SLP.npz' else m0
                rows.append(dict(arm=a, zfile=os.path.basename(fz), k=k, Mused=int(m), **{q: r.get(q) for q in ('id', 'unit', 'state', 'agent', 'animal', 'M')}))
                jobs.append(((a, str(wid), m), k, [k]))
        Wd = {a: np.load(f'{d}/windows.npz') for a, d in src.items()}
        load = lambda t, k: ns(Wd[t[0]][t[1]].astype(float)[:, S(t[2])]); key = lambda r: r['unit']
        edges = {}
        for u in dict.fromkeys(r['unit'] for r in rows):
            ws = [load(j[0], 0) for r, j in zip(rows, jobs) if r['unit'] == u and r['state'] == 'awake_ec']
            if ws: edges[u] = top_edges(ws)
    keep = [n for n, r in enumerate(rows) if key(r) in edges]; rows = [rows[n] for n in keep]; jobs = [jobs[n] for n in keep]
    F = Parallel(n_jobs=7, verbose=2)(delayed(feats)(load(a, b), edges[key(r)], s) for (a, b, s), r in zip(jobs, rows))
    np.savez(out, F=np.array(F), rows=json.dumps(rows))
    print('done', out, np.array(F).shape, flush=True)
