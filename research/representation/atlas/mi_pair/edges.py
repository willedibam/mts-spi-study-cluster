"""Per-edge Gaussian and kNN MI, real and joint IAAFT, on the NeuroTycho 10-s atlas windows.

recompute: re-runs the lost `zns_[AB]_<agent>.npz` exactly as review/zgk_ns.py did (32-channel subset,
           raw and normal-scored, one spectrum-final IAAFT draw seeded by the row index within the per-agent
           call), and also stores every edge's Gaussian and kNN MI.
           python edges.py recompute <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
full:      all kept channels, normal-scored, three joint IAAFT draws per window; rows sharded i of n.
           python edges.py full <bank dir> <out.npz> <stage tag> <shard i> <n shards> <n workers>
Rules and decision criteria: edges-prereg.md (frozen before any output is read).
"""
import sys, json, numpy as np, warnings
from multiprocessing import Pool
from scipy.special import digamma
from scipy.stats import norm, rankdata
from mtsspi.surrogates import multivariate_iaaft
warnings.filterwarnings('ignore')
K_NN, STATES, B_FULL = 4, ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec'), 3


def acf(x):  # verbatim from review/surr_pairs.py
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n); a = np.fft.irfft(f * np.conj(f))[:n]; return a / a[0]


def ksg(x, y, w):  # verbatim from review/surr_pairs.py
    n = len(x); dx = np.abs(x[:, None] - x[None, :]); dy = np.abs(y[:, None] - y[None, :]); d = np.maximum(dx, dy)
    t = np.arange(n); ex = np.abs(t[:, None] - t[None, :]) <= w
    d = np.where(ex, np.inf, d); eps = np.sort(d, 1)[:, K_NN - 1]
    nx = np.sum(np.where(ex, False, dx < eps[:, None]), 1); ny = np.sum(np.where(ex, False, dy < eps[:, None]), 1)
    return digamma(K_NN) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))


def ksg_rows(x, i, js, w, tdist):
    """ksg(x[:, i], x[:, j], w[j]) for every j in js, reusing |x_i(t) - x_i(s)|. Same arithmetic as ksg."""
    dx = np.abs(x[:, i][:, None] - x[:, i][None, :]); out = np.empty(len(js)); n = len(x)
    for a, j in enumerate(js):
        dy = np.abs(x[:, j][:, None] - x[:, j][None, :]); ex = tdist <= w[a]
        d = np.where(ex, np.inf, np.maximum(dx, dy)); eps = np.sort(d, 1)[:, K_NN - 1]
        nx = np.sum(np.where(ex, False, dx < eps[:, None]), 1); ny = np.sum(np.where(ex, False, dy < eps[:, None]), 1)
        out[a] = digamma(K_NN) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return out


def edge_mi(x):
    """Gaussian MI from r and KSG MI (AUTO Theiler window) for every i < j, on standardised x, as zgk_ns.zgk."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; A = np.array([acf(x[:, m]) for m in range(M)])
    W = np.array([[min(int(2 * np.dot(A[i], A[j])), T // 10) for j in range(M)] for i in range(M)]); t = np.arange(T); tdist = np.abs(t[:, None] - t[None, :])
    k = np.concatenate([ksg_rows(x, i, np.arange(i + 1, M), W[i, i + 1:], tdist) for i in range(M - 1)])
    return -0.5 * np.log(1 - r ** 2), k


ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))
zc = lambda g, k: np.corrcoef(g, k)[0, 1]
sub32 = lambda m: np.round(np.arange(32) * (m - 1) / 31).astype(int)


def one_recompute(args):
    x, seed = args; V, E = [], []
    for v in (x, ns(x)):
        s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
        gr, kr = edge_mi(v); gs, ks = edge_mi(s)
        V.append([zc(gr, kr), zc(gs, ks)]); E.append([[gr, kr], [gs, ks]])
    return np.array(V), np.array(E, np.float32)


def one_full(args):
    x, seed = args; v = ns(x); g, k = edge_mi(v); G, K = [], []
    for b in range(B_FULL):
        s = multivariate_iaaft(v, np.random.default_rng(seed + b), final='spectrum')[0]
        gs, ks = edge_mi(s); G.append(gs); K.append(ks)
    return np.array([g, k, np.mean(G, 0), np.mean(K, 0), np.std(K, 0)], np.float32)


def meta(rows, man):
    kept = {u['unit']: u['kept_channels'] for u in man['preprocessing']['units']}
    return dict(delta_fraction=np.array([r.get('delta_fraction', np.nan) for r in rows], float),
                **{k: np.array([r.get(k, '') for r in rows]) for k in ('id', 'unit', 'state', 'agent', 'animal')}), kept


if __name__ == '__main__':
    mode, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES]
    if mode == 'recompute':
        agent, rowmode, nw = sys.argv[4], sys.argv[5], int(sys.argv[6])
        rows = [r for r in rows if r['agent'] in agent.split(',')]
        S0 = sub32(rows[0]['M'])
        sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
        jobs = [(W[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)]
        with Pool(nw) as p: res = p.map(one_recompute, jobs, chunksize=1)
        m, kept = meta(rows, man)
        labels = np.array([np.array(kept[r['unit']][:r['M']])[sub(r)] for r in rows])  # electrode numbers of the 32
        np.savez(out, V=np.array([a for a, _ in res]), E=np.array([b for _, b in res]), chan=labels, rowmode=rowmode, **m)
    elif mode == 'full':
        tag, i, n, nw = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        idx = [g for g in range(len(rows)) if g % n == i]
        seed0 = {'A': 0, 'B': 500000}[tag]
        jobs = [(W[rows[g]['id']].astype(float), seed0 + 1000 * g) for g in idx]
        with Pool(nw) as p: res = p.map(one_full, jobs, chunksize=1)
        m, kept = meta([rows[g] for g in idx], man)
        Xo = np.empty(len(res), dtype=object)
        for q, a in enumerate(res): Xo[q] = a  # per-window (5, E); E differs between 126- and 127-channel units
        np.savez(out, X=Xo, row=np.array(idx), M=np.array([rows[g]['M'] for g in idx]),
                 chan=np.array([json.dumps(kept[rows[g]['unit']][:rows[g]['M']]) for g in idx]), **m)
    print('done', mode, out, flush=True)
