"""Post-hoc secondary (kshift): the completed MI-pair drug tests rescored with shift-corrected KSG, side by side with k4.

Per window, each study's own preprocessing, channels and joint-IAAFT seeds (so the k4 column must reproduce the stored z),
then per pair, on the real window and on every surrogate:
  g       Gaussian MI −½ log(1 − r²)
  k4      KSG k = 4, AUTO Theiler window w = min(int(2 acf_i·acf_j), T//10), ψ(N)    [frozen primary estimator]
  kshift  k4 minus the mean k4 of (x_i, x_j circularly shifted by T/4, T/2, 3T/4), same w and ψ(N)   (ksgbias.py edge_all)
  k2te    KSG with w = min(2 max(τ_e,i, τ_e,j), T//10) and pool ψ (ksgbias.py k2te); optional candidate, not a primary
z_e = Pearson corr over pairs of (g, K_e). KSG is the KD-tree `claar/surr_pairs.ksg` (identical output to brute force).

    python kz.py claar  SEL_DIR OUT.npz NJOBS            (claar_z.py variant ns: CAR, last dropped, normal-scored, 3 draws)
    python kz.py kauvar SEL_DIR OUT.npz NJOBS            (claar_z.py variant asis: normal-scored, 3 draws)
    python kz.py prios  WIN_DIR OUT.npz NJOBS [all10]    (human/score.py car variant; all10 = every good channel, 10 draws)
    python kz.py gate   STUDY SEL_DIR STORED.npz N [OPT] (k4 only: z of N spread windows against the stored z, real and surrogate)
    python kz.py check                                   (per-edge k4, kshift, k2te against ksgbias.edge_all)
OPT nok2te skips k2te. Out: Z[window, estimator (k4, kshift, k2te), (real, surr1, ...)]; O[window, (real, surr...)] the
mean over pairs of the subtracted shift offset k4 − kshift; rows (json); est.
"""
import sys, os, glob, json, zlib, numpy as np, warnings
from joblib import Parallel, delayed
from scipy.special import digamma
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(MP, 'claar'), os.path.join(MP, 'human'), MP, os.path.join(MP, '../../../..')]
from surr_pairs import acf, ksg  # noqa: E402  (KD-tree KSG, K_NN = 4)
from mtsspi.surrogates import multivariate_iaaft  # noqa: E402
from zgk_ns import ns  # noqa: E402
from research.representation.atlas.prepare import reference  # noqa: E402
warnings.filterwarnings('ignore')
EST = ('k4', 'kshift', 'k2te')


def tau_e(a):
    b = np.nonzero(a < 1 / np.e)[0]; return int(b[0]) if len(b) else len(a)


def pool_psi(N, w):
    i = np.arange(N); return np.mean(digamma(N - np.minimum(i, w) - np.minimum(N - 1 - i, w)))


def edges(x, ests=EST):
    """Per pair g and K_e (rows in EST order; NaN where e not in ests) on one window (T, M); k4 arithmetic of zgk_ns.zgk."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = [acf(x[:, m]) for m in range(M)]; te = [tau_e(a) for a in A]
    K = np.full((3, len(r)), np.nan)
    for p, (i, j) in enumerate(zip(*iu)):
        w = min(int(2 * np.dot(A[i], A[j])), T // 10); K[0, p] = ksg(x[:, i], x[:, j], w)
        if 'kshift' in ests: K[1, p] = K[0, p] - np.mean([ksg(x[:, i], np.roll(x[:, j], q * T // 4), w) for q in (1, 2, 3)])
        if 'k2te' in ests:
            w2 = min(2 * max(te[i], te[j]), T // 10); K[2, p] = ksg(x[:, i], x[:, j], w2) - digamma(T) + pool_psi(T, w2)
    return -0.5 * np.log(1 - r ** 2), K


def zall(x, ests=EST):
    """z_e for e in EST, plus the mean shift offset k4 − kshift over pairs (the subtracted independence bias)."""
    g, K = edges(x, ests)
    return [np.corrcoef(g, k)[0, 1] if np.isfinite(k).all() else np.nan for k in K] + [np.mean(K[0] - K[1])]


def one(v, seed, nd, ests=EST):
    """(4: z_k4, z_kshift, z_k2te, mean offset; 1 + nd): real then surrogate draws seed + d, as claar_z.one / human/score.one."""
    out = [zall(v, ests)] + [zall(multivariate_iaaft(v, np.random.default_rng(seed + d), final='spectrum')[0], ests) for d in range(nd)]
    return np.array(out).T


def jobs_sel(src):
    """claar_z.py job list: sorted sub-*.npz, seeds crc32('<session>:<i>') * 10."""
    J, rows = [], []
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        Z = np.load(fn); R = json.loads(str(Z['rows'])); s = os.path.basename(fn)[:-4]
        for i, r in enumerate(R):
            rows.append(dict(r, session=s)); J.append((fn, i, zlib.crc32(f'{s}:{i}'.encode()) * 10))
    return J, rows


def prep_claar(w): return ns(reference(w.astype(np.float64), 'car'))
def prep_kauvar(w): return ns(w.astype(np.float64))


def prep_prios(w, stim, every):
    """human/score.one, car variant: stimulated electrodes leave the CAR and the analysed set (Deviation 1)."""
    from score import channels
    w = w.astype(float); keep = np.isin(np.arange(w.shape[1]), stim, invert=True)
    idx = [j for j in channels(w.shape[1], every) if keep[j]]
    return ns((w - w[:, keep].mean(1, keepdims=True))[:, idx])


def tasks(study, src, opt=''):
    """[(prepared window loader, seed, n draws)], rows."""
    if study in ('claar', 'kauvar'):
        J, rows = jobs_sel(src); prep = prep_claar if study == 'claar' else prep_kauvar
        return [((lambda fn=fn, i=i: prep(np.load(fn)['W'][i])), seed, 3) for fn, i, seed in J], rows
    al = [p for p in opt.split(',') if p.startswith('all')]; nd, every = (int(al[0][3:] or 3), True) if al else (3, False)
    T, rows = [], []
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        Z = np.load(fn); s = os.path.basename(fn)[:-4]
        for i, r in enumerate(json.loads(str(Z['rows']))):
            rows.append(dict(r, subject=s))
            T.append(((lambda fn=fn, i=i, st=r.get('stim', []): prep_prios(np.load(fn)['W'][i], st, every)),
                      zlib.crc32(f'{s}:{i}'.encode()) * 10, nd))
    return T, rows


def run(study, src, out, nj, opt=''):
    T, rows = tasks(study, src, opt)
    ests = tuple(e for e in EST if e != 'k2te' or 'nok2te' not in opt)
    res = np.array(Parallel(n_jobs=nj, verbose=5, batch_size=1)(delayed(one)(load(), seed, nd, ests) for load, seed, nd in T), np.float64)
    np.savez(out, Z=res[:, :3], O=res[:, 3], rows=json.dumps(rows), est=np.array(EST)); print('done', out, len(rows), flush=True)


def gate(study, src, stored, n, opt=''):
    """k4 real and surrogate z of windows 0..n-1 (k2te skipped) against the stored study z."""
    T, rows = tasks(study, src, opt); d = np.load(stored)
    S = d['Z'][:, 0] if 'Z' in d.files else d['ns' if study == 'claar' else 'asis']
    idx = np.unique(np.linspace(0, len(T) - 1, n).astype(int)); assert len(S) == len(T), (len(S), len(T))
    got = np.array(Parallel(n_jobs=min(len(idx), 8))(delayed(one)(T[i][0](), T[i][1], T[i][2], ('k4',)) for i in idx))[:, 0]
    gr, gs = np.abs(got[:, 0] - S[idx, 0]).max(), np.abs(got[:, 1:] - S[idx, 1:]).max()
    print(f'gate {study}: windows {idx.tolist()}, max |k4 z − stored| real {gr:.2e}, surrogate {gs:.2e} ->', 'PASS' if max(gr, gs) < 1e-6 else 'FAIL')


def check_ref(seed=0, T=300, M=5):
    """Per-edge k4, kshift and k2te equal ksgbias.edge_all on a small synthetic window (brute-force reference)."""
    from ksgbias import edge_all
    rng = np.random.default_rng(seed); x = np.cumsum(rng.standard_normal((T, M)), 0) * .05 + rng.standard_normal((T, M))
    x[:, 1] += .6 * x[:, 0]; g, K = edges(x); R = edge_all(x)  # R rows: g, k4, kpool, k2te, kshift, w
    gap = max(np.abs(g - R[0]).max(), np.abs(K[0] - R[1]).max(), np.abs(K[1] - R[4]).max(), np.abs(K[2] - R[3]).max())
    print(f'reference check vs ksgbias.edge_all ({M * (M - 1) // 2} edges, T {T}): max gap {gap:.2e} ->', 'PASS' if gap < 1e-9 else 'FAIL')


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'check': check_ref()
    elif a[1] == 'gate': gate(a[2], a[3], a[4], int(a[5]), a[6] if len(a) > 6 else '')
    else: run(a[1], a[2], a[3], int(a[4]), a[5] if len(a) > 5 else '')
