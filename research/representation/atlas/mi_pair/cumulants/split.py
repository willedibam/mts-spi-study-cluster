"""Post hoc (23 November, after the frozen cumulant result): where in the fourth-order term does the drug change sit, and why did
the earlier propofol hub test (../review/mech.py, hubtest.py) read joint co-kurtosis ~ 0?

Per edge, the pair is whitened in its own principal axes, u = (a + b) / sqrt(2(1 + r)) and v = (a - b) / sqrt(2(1 - r)).
J4 (rotation-invariant) splits into
  Ksum  k40^2 / 48                         kurtosis along the shared direction
  Kdiff k04^2 / 48                         kurtosis along the difference direction (pair coming apart in bursts)
  Kmix  (4 k31^2 + 6 k22^2 + 4 k13^2) / 48 magnitude co-modulation of shared and difference parts
plus mech.py's two readouts on the unwhitened pair: cok = mean(a^2 b^2) - (1 + 2 r^2) and rk = residual excess kurtosis
(both regression directions). Levels are means over edges, net of the anchor's own IAAFT draw; 'all' edges and the top
10% by real |r| ('tight', the hub test's population).
  python split.py calib <n seeds> <n workers> <out.npz>
  python split.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>     (same rows and seeds as run.py)
  python split.py score <dir with split_[AB]_*.npz>
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')
P = ('Ksum', 'Kdiff', 'Kmix', 'J4', 'cok', 'rk')
KEYS = tuple(f'{p}_{e}' for e in ('all', 'tight') for p in P)


def split4(x):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu, ju = np.triu_indices(M, 1)
    r = (x.T @ x / T)[iu, ju]; a, b = x[:, iu], x[:, ju]
    u, v = (a + b) / np.sqrt(2 * (1 + r)), (a - b) / np.sqrt(2 * (1 - r))
    m = lambda p, q: np.mean(u ** p * v ** q, 0); m20, m02, m11 = m(2, 0), m(0, 2), m(1, 1)
    k40, k31 = m(4, 0) - 3 * m20 ** 2, m(3, 1) - 3 * m20 * m11
    k22, k13, k04 = m(2, 2) - m20 * m02 - 2 * m11 ** 2, m(1, 3) - 3 * m02 * m11, m(0, 4) - 3 * m02 ** 2
    Ks, Kd, Km = k40 ** 2 / 48, k04 ** 2 / 48, (4 * k31 ** 2 + 6 * k22 ** 2 + 4 * k13 ** 2) / 48
    cok = np.mean(a ** 2 * b ** 2, 0) - (1 + 2 * r ** 2); rk = 0
    for p, q in ((a, b), (b, a)):
        e = q - r * p; e = (e - e.mean(0)) / e.std(0); rk = rk + (np.mean(e ** 4, 0) - 3) / 2
    return np.array([Ks, Kd, Km, Ks + Kd + Km, cok, rk]), np.abs(r)


def net(v, s):
    (R, ar), (S, _) = split4(v), split4(s); t = ar >= np.quantile(ar, .9); D = R - S
    return np.r_[D.mean(1), D[:, t].mean(1)]


def calib(B, nw, out):
    import time
    from joblib import Parallel, delayed
    import calib as cc
    def one(kind, seed):
        rng = np.random.default_rng(seed); v = cc.ns(cc.synth(kind, rng)); return net(v, cc.n0(v, np.random.default_rng([seed, 0])))
    t = time.time(); R = {}; print('kind          ' + ' '.join(f'{k:>12s}' for k in KEYS))
    for kind in cc.KINDS:
        R[kind] = np.array(Parallel(n_jobs=nw)(delayed(one)(kind, 1000 * (cc.KINDS.index(kind) + 1) + s) for s in range(B)))
        m, sd = R[kind].mean(0), R[kind].std(0, ddof=1)
        print(f'{kind:13s} ' + ' '.join(f'{a:+.4f}({a / b * np.sqrt(B):+4.0f})' for a, b in zip(m, sd)), f'[{time.time() - t:.0f} s]', flush=True)
    np.savez(out, keys=KEYS, **R)


def one(args):
    from run import ns
    from mtsspi.surrogates import multivariate_iaaft
    x, seed = args; v = ns(x); return net(v, multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0])


def score(d):
    from scipy.stats import ttest_1samp
    Q = [np.load(f) for f in sorted(glob.glob(f'{d}/split_[AB]_*.npz'))]; cat = lambda k: np.concatenate([q[k] for q in Q])
    W, unit, state, agent, animal = cat('C'), cat('unit'), cat('state'), cat('agent'), cat('animal'); out = {}
    for c, nm in enumerate(KEYS):
        S = []
        for u in np.unique(unit):
            ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'; m = lambda st: (unit == u) & (state == st)
            if m(s).any() and m('awake_ec').any(): S.append((ag, animal[unit == u][0], W[m(s), c].mean() - W[m('awake_ec'), c].mean(), W[m('awake_ec'), c].mean()))
        ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); dz, aw = np.array([q[2] for q in S]), np.array([q[3] for q in S]); dr = ag != 'SLP'
        A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])]); t = ttest_1samp(A, 0)
        agm = {a: float(dz[ag == a].mean()) for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        out[nm] = dict(awake=float(aw[dr].mean()), delta=float(dz[dr].mean()), animals=A.tolist(), t=float(t.statistic), agents=agm)
        print(f'{nm:11s} awake {aw[dr].mean():+.5f} Δ {dz[dr].mean():+.5f} animals {np.round(A, 5)} t(3) {t.statistic:+.2f} | '
              + ' '.join(f'{a} {v:+.5f}' for a, v in agm.items()))
    for e in ('all', 'tight'):
        J = out[f'J4_{e}']['delta']; print(f'{e}: share of ΔJ4 ' + ' '.join(f'{p} {out[f"{p}_{e}"]["delta"] / J:.2f}' for p in ('Ksum', 'Kdiff', 'Kmix')))
    Path(d, 'split-score.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'calib': calib(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]); sys.exit()
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    from multiprocessing import Pool
    from edges import sub32, meta, STATES
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: C = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); np.savez(out, C=np.array(C), **m); print('done', out, flush=True)
