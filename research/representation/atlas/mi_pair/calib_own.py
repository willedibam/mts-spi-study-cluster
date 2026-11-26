"""Stationary linear-Gaussian calibration of the residual's drug − awake change at a dataset's own state spectra
(rules: preregs/speccalib-prereg.md).

    python calib_own.py prios <human/prios windows dir> <z.npz> <out.npz> <workers>
    python calib_own.py ferraina <sel dir> <fb_z_secA.npz> <out.npz> <workers>
    python calib_own.py score <out.npz> <observed Δ> [<observed Δ, descriptive> ...]
"""
import sys, os, json, glob, numpy as np, warnings
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, 'claar')]
from zgk_ns import zgk, ns
from surr_pairs import multivariate_iaaft
from ksgbias_calib import phase_rand, markov
warnings.filterwarnings('ignore')
NB_DRAWS = 3
COUNTS = dict(prios=(80, 20), ferraina=(150, 40))  # block, gain_shared per subject-state (PRIOS) or per state (monkey B)


def gen(kind, src, rng):
    """ksgbias_calib.gen with every source channel (M = src.shape[1]) and M // 4 coupled blocks of four."""
    T, M = src.shape; X = phase_rand(src[:, rng.permutation(M)], rng); L = np.eye(M)
    for b in range(M // 4):
        idx = np.arange(b * 4, b * 4 + 4); C = np.full((4, 4), rng.uniform(.2, .9)); np.fill_diagonal(C, 1)
        C = C + np.diag(rng.uniform(0, .3, 4)); d = np.sqrt(np.diag(C)); L[np.ix_(idx, idx)] = np.linalg.cholesky(C / np.outer(d, d))
    X = X @ L.T
    if kind == 'gain_shared': X = X * np.where(markov(T, T / 10, rng), 3., 1.)[:, None]
    return X


def one(a):
    kind, seed, src = a; x = ns(gen(kind, src, np.random.default_rng(seed)))
    return [zgk(x)] + [zgk(multivariate_iaaft(x, np.random.default_rng(seed + 1 + d), final='spectrum')[0]) for d in range(NB_DRAWS)]


def prios_sources(d, zf):
    """score.py's car variant: CAR over good non-stimulated channels, then channels(m) without stimulated ones; ns."""
    sys.path.insert(0, os.path.join(HERE, 'human')); from score import channels
    Z = np.load(zf); zrows = json.loads(str(Z['rows'])); out, gate, k = {}, [], 0
    for fn in sorted(glob.glob(f'{d}/sub-*.npz')):
        F = np.load(fn); s = os.path.basename(fn)[:-4]; rows = json.loads(str(F['rows']))
        for i, r in enumerate(rows):
            w = F['W'][i].astype(float); keep = np.isin(np.arange(w.shape[1]), r.get('stim', []), invert=True)
            idx = [j for j in channels(w.shape[1]) if keep[j]]; v = ns((w - w[:, keep].mean(1, keepdims=True))[:, idx])
            assert zrows[k]['subject'] == s; out.setdefault(s, {}).setdefault(r['state'], []).append(v)
            if i in (0, len(rows) - 1) and s == 'sub-PRIOS01': gate.append((zgk(v), float(Z['Z'][k, 0, 0])))
            k += 1
    return {s: v for s, v in out.items() if min(len(v.get(st, [])) for st in ('awake', 'anaes')) >= 20}, gate


def ferraina_sources(d, zf):
    acc = json.load(open(f'{d}/sel.json'))['accepted']; W = np.load(f'{d}/W32pa.npy', mmap_mode='r'); Zpa = np.load(zf)['Z_pa']
    st = np.array([r['state'] for r in acc]); out = {'W1': [], 'D': []}
    for i in np.nonzero(np.isin(st, ('W1', 'D')))[0]: out[st[i]].append(ns(W[i].astype(float)))
    gate = [(zgk(ns(W[i].astype(float))), float(Zpa[i, 0])) for i in (0, int(np.nonzero(st == 'D')[0][0]))]
    return {'monkeyB': out}, gate


def build(src, nb, ng, base):
    jobs, lab = [], []
    for si, (s, states) in enumerate(sorted(src.items())):
        for ti, (t, wins) in enumerate(sorted(states.items())):
            for ki, (kind, n) in enumerate((('block', nb), ('gain_shared', ng))):
                for i in range(n):
                    jobs.append((kind, base + 10 ** 6 * si + 10 ** 5 * ti + 10 ** 4 * ki + 10 * i, wins[int(i * len(wins) / n)]))
                    lab.append((s, t, kind))
    return jobs, lab


def score(f, obs):
    D = np.load(f); Z, lab = D['Z'], D['lab']; res = Z[:, 0] - Z[:, 1:].mean(1)
    print(f'gate (stored real z vs zgk here): {D["gate"].tolist()}')
    drug, awake = (('anaes', 'awake') if 'anaes' in lab[:, 1] else ('D', 'W1')); O, V, det = [], [], {}
    for s in np.unique(lab[:, 0]):
        g = lambda t, k: res[(lab[:, 0] == s) & (lab[:, 1] == t) & (lab[:, 2] == k)]
        bd, bw = g(drug, 'block'), g(awake, 'block'); O.append(bd.mean() - bw.mean()); V.append(bd.var(ddof=1) / len(bd) + bw.var(ddof=1) / len(bw))
        for t in (awake, drug): det[(s, t)] = (g(t, 'block').mean() - g(t, 'gain_shared').mean()) / g(t, 'block').std(ddof=1)
        print(f'  {s}: block residual {awake} {bw.mean():+.4f} (sd {bw.std(ddof=1):.4f}, n {len(bw)}), {drug} {bd.mean():+.4f} (sd {bd.std(ddof=1):.4f}); '
              f'offset {O[-1]:+.4f}; real z block {awake}/{drug} {np.mean(Z[(lab[:, 0] == s) & (lab[:, 1] == awake) & (lab[:, 2] == "block"), 0]):.3f}/'
              f'{np.mean(Z[(lab[:, 0] == s) & (lab[:, 1] == drug) & (lab[:, 2] == "block"), 0]):.3f}; detectability {det[(s, awake)]:.2f}/{det[(s, drug)]:.2f}')
    O_, se = np.mean(O), np.sqrt(np.sum(V)) / len(O); d0 = obs[0]
    print(f'offset O {O_:+.4f} ± {se:.4f} (SE); observed {d0:+.4f}; net {d0 - O_:+.4f}; (Δ_obs − O) / SE(O) {(d0 - O_) / se:+.2f}')
    print('PRIMARY:', 'DISTINGUISHED' if d0 - O_ < -2 * se else 'NOT DISTINGUISHED', '(O ≥ 0: conservative)' if O_ >= 0 else '')
    for o in obs[1:]: print(f'descriptive: observed {o:+.4f}, net {o - O_:+.4f}, {(o - O_) / se:+.2f} SE')
    print(f'detectability < .5 in every state-subject: {all(v < .5 for v in det.values())}')
    return np.array(O)


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'score':
        score(sys.argv[2], [float(x) for x in sys.argv[3:]]); sys.exit()
    d, zf, out, nw = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    src, gate = (prios_sources if m == 'prios' else ferraina_sources)(d, zf); print('gate', gate, flush=True)
    jobs, lab = build(src, *COUNTS[m], base={'prios': 8 * 10 ** 7, 'ferraina': 9 * 10 ** 7}[m])
    print(m, len(jobs), 'windows', {s: {t: len(w) for t, w in v.items()} for s, v in src.items()}, flush=True)
    with Pool(nw) as p: Z = p.map(one, jobs, chunksize=4)
    np.savez(out, Z=np.array(Z), lab=np.array(lab), gate=np.array(gate)); print('done', out, flush=True)
