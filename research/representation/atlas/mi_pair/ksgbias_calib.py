"""Synthetic calibration for ksgbias.py: KSG bias at edges of known MI, real against its own IAAFT draw, per estimator.

Channels are independent Gaussian processes with the periodograms of real normal-scored macaque channels (drug PF, drug KTMD
or awake eyes-closed windows of stage A, 32-channel subset as edges.py, 16 channels drawn per synthetic window), so true MI
is 0 between blocks. Generators (M = 16, four blocks of four):
  indep        no coupling: every edge has MI 0
  block        stationary linear coupling within blocks (random strength); between-block edges MI 0
  gain_ind     block + independent two-state gain per channel (x3, mean dwell 1 s): between-block edges still MI 0
  burst_ind    block + independent gain per channel (x5, mean dwell 0.5 s): between-block edges still MI 0
  gain_shared  block + one shared gain (x3, dwell 1 s): real non-Gaussian dependence on every edge (sensitivity, not size)
Each is normal-scored, then given its own spectrum-final joint IAAFT draw, and scored with ksgbias.edge_all (the exact code
used on the real windows). T = 1000 (10-s arm); a T = 200 variant (first 200 samples of the source) for the 2-s arm.

  python ksgbias_calib.py <stage-a bank dir> <out.npz> <n per generator> <n workers>
  python ksgbias_calib.py score <out.npz>
  python ksgbias_calib.py mech <stage-a bank dir> <out.npz> <n> <n workers>
mech: stationary independent pairs (MI 0) only, KSG k=4 with pool psi at w = 0, AUTO, 2, 4, 8 x AUTO (capped T/4), plus the
same pair with time order shuffled (iid, w = 0), and k = 10 at AUTO: where the independence bias comes from.
"""
import sys, json, numpy as np, warnings
from multiprocessing import Pool
from mtsspi.surrogates import multivariate_iaaft
from edges import ns, zc, sub32, STATES
from ksgbias import edge_all, EST
warnings.filterwarnings('ignore')
M, NB = 16, 4
KINDS = ('indep', 'block', 'gain_ind', 'burst_ind', 'gain_shared')
SETS = (('PF', 1000), ('KTMD', 1000), ('AW', 1000), ('PF', 200))


def phase_rand(X, rng):
    T = len(X); F = np.fft.rfft(X, axis=0); ph = np.exp(2j * np.pi * rng.random(F.shape)); ph[0] = 1
    if T % 2 == 0: ph[-1] = 1
    Y = np.fft.irfft(np.abs(F) * ph, T, axis=0); return (Y - Y.mean(0)) / Y.std(0)


def markov(T, dwell, rng):
    flip = rng.random(T) < 1 / dwell; flip[0] = rng.random() < .5; return np.cumsum(flip) % 2


def gen(kind, src, rng):
    T = len(src); X = phase_rand(src[:, rng.permutation(src.shape[1])[:M]], rng)
    if kind != 'indep':
        L = np.eye(M)
        for b in range(NB):
            idx = np.arange(b * 4, b * 4 + 4); C = np.full((4, 4), rng.uniform(.2, .9)); np.fill_diagonal(C, 1)
            C = C + np.diag(rng.uniform(0, .3, 4)); d = np.sqrt(np.diag(C))
            L[np.ix_(idx, idx)] = np.linalg.cholesky(C / np.outer(d, d))
        X = X @ L.T
    if kind == 'gain_ind': X = X * np.where(np.array([markov(T, T / 10, rng) for _ in range(M)]).T, 3., 1.)
    if kind == 'burst_ind': X = X * np.where(np.array([markov(T, T / 20, rng) for _ in range(M)]).T, 5., 1.)
    if kind == 'gain_shared': X = X * np.where(markov(T, T / 10, rng), 3., 1.)[:, None]
    return X


def one(a):
    kind, seed, src = a; rng = np.random.default_rng(seed)
    x = ns(gen(kind, src, rng)); s = multivariate_iaaft(x, np.random.default_rng(seed + 1), final='spectrum')[0]
    return np.array([edge_all(x), edge_all(s)], np.float32)


def mech_one(a):
    from ksgbias import kcore, pool_psi, tau_e
    from scipy.special import digamma
    seed, src = a; rng = np.random.default_rng(seed); x = ns(gen('indep', src, rng)); T = len(x)
    from edges import acf
    A = np.array([acf(x[:, m]) for m in range(M)]); t = np.arange(T); td = np.abs(t[:, None] - t[None, :]); out = []
    for i, j in zip(*np.triu_indices(M, 1)):
        w = min(int(2 * A[i] @ A[j]), T // 10); dx = np.abs(x[:, i][:, None] - x[:, i][None, :]); dy = np.abs(x[:, j][:, None] - x[:, j][None, :])
        row = [kcore(dx, dy, td <= min(m * w, T // 4) if m else td <= 0) + pool_psi(T, min(m * w, T // 4) if m else 0) for m in (0, 1, 2, 4, 8)]
        p = rng.permutation(T); xs, ys = x[p, i], x[rng.permutation(T), j]
        row.append(kcore(np.abs(xs[:, None] - xs[None, :]), np.abs(ys[:, None] - ys[None, :]), td <= 0) + digamma(T))
        d = np.where(td <= w, np.inf, np.maximum(dx, dy)); eps = np.partition(d, 9, 1)[:, 9]; ex = td <= w
        nx = np.sum(~ex & (dx < eps[:, None]), 1); ny = np.sum(~ex & (dy < eps[:, None]), 1)
        row.append(digamma(10) + pool_psi(T, w) - np.mean(digamma(nx + 1) + digamma(ny + 1))); row.append(w); out.append(row)
    return np.array(out)


def mech(d, out, n, nw):
    src = sources(d); jobs = [(9 * 10 ** 6 + 100 * si + i, src[tag][i % len(src[tag])][:T]) for si, (tag, T) in enumerate(SETS) for i in range(n)]
    lab = np.array([f'{tag}-T{T}' for tag, T in SETS for _ in range(n)])
    with Pool(nw) as p: R = np.array(p.map(mech_one, jobs, chunksize=1))
    np.savez(out, R=R, set=lab); names = ('w=0', 'AUTO', '2xAUTO', '4xAUTO', '8xAUTO', 'shuffled iid', 'k10 AUTO')
    for s in np.unique(lab):
        q = R[lab == s]; print(f'{s}: median AUTO w {np.median(q[..., -1]):.0f}; mean K at MI 0 (sd):',
                              ' '.join(f'{nm} {q[..., c].mean():+.4f} ({q[..., c].std():.3f})' for c, nm in enumerate(names)))


def sources(d):
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz'); out = {}
    for tag, ag, st in (('PF', 'PF', 'drug'), ('KTMD', 'KTMD', 'drug'), ('AW', 'KTMD', 'awake_ec')):
        rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] == ag]; S0 = sub32(rows[0]['M'])
        out[tag] = [ns(Wn[r['id']].astype(float)[:, S0]) for r in rows if r['state'] == st]
    return out


def score(f):
    D = np.load(f); X, kind, sset = D['X'].astype(float), D['kind'], D['set']
    iu = np.triu_indices(M, 1); between = (iu[0] // 4) != (iu[1] // 4)
    for s in np.unique(sset):
        print(f'\n=== {s}')
        for k in KINDS:
            q = (sset == s) & (kind == k); A = X[q]  # (n, real/surr, 6, E)
            print(f'{k} (n {q.sum()}, median w {np.median(A[:, 0, 5]):.0f}, at cap {(A[:, 0, 5] >= int(s.split("-T")[1]) // 10).mean():.2f})')
            for c, e in enumerate(EST):
                kr, ks, g, gs = A[:, 0, 1 + c], A[:, 1, 1 + c], A[:, 0, 0], A[:, 1, 0]
                dz = np.array([zc(g[i], kr[i]) - zc(gs[i], ks[i]) for i in range(len(A))]) if k != 'indep' else np.full(1, np.nan)
                print(f'  {e:6s} MI-0 edges: K real {kr[:, between].mean():+.4f}, surr {ks[:, between].mean():+.4f}, '
                      f'real − surr {np.mean(kr[:, between] - ks[:, between]):+.4f} (sd real {kr[:, between].std():.4f}) | '
                      f'coupled K − G real {np.mean((kr - g)[:, ~between]):+.4f}, surr {np.mean((ks - gs)[:, ~between]):+.4f} | '
                      f'z real − surr {dz.mean():+.4f} ± {dz.std(ddof=1) / np.sqrt(len(dz)) if len(dz) > 1 else np.nan:.4f}')


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    if sys.argv[1] == 'mech': mech(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])); sys.exit()
    d, out, n, nw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); src = sources(d); jobs, lab = [], []
    for si, (tag, T) in enumerate(SETS):
        for ki, k in enumerate(KINDS):
            for i in range(n):
                seed = 10 ** 6 * si + 10 ** 4 * ki + 2 * i
                jobs.append((k, seed, src[tag][i % len(src[tag])][:T])); lab.append((k, f'{tag}-T{T}'))
    with Pool(nw) as p: R = p.map(one, jobs, chunksize=1)
    np.savez(out, X=np.array(R), kind=np.array([a for a, _ in lab]), set=np.array([b for _, b in lab]))
    print('done', out, flush=True); score(out)
