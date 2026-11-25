"""Extra joint IAAFT draws for the off-band count's null (preregs/offband-prereg.md, step 2), K only.

The windows, 32-channel subsets, normal-scoring and per-agent row indices k are those of `edges.py recompute` as invoked
by jobs/edges_recompute.pbs; draw 0 is default_rng(k), so it is the stored zns surrogate, and draw b >= 1 is
default_rng([k, b]), a stream disjoint from every row's draw 0. G is not stored: spectrum-final IAAFT keeps r, so each
draw's G is the recording's. Stored per window: K of each draw (float32), and the recording's and draw 0's Theiler
windows (equal if the spectrum is kept).

    python offband_null.py gate  <banks> <zns dir> <n windows> <n workers>          draw 0 of the first windows of zns_A_PF
    python offband_null.py tier1 <banks> <out.npz> <shard i> <n shards> <n draws> <n workers>
    python offband_null.py tier2 <banks> <out.npz> <window id> <n draws> <block> <n workers>
"""
import sys, json, numpy as np, warnings
from multiprocessing import Pool
from mtsspi.surrogates import multivariate_iaaft
from edges import acf, edge_mi, ns, sub32, STATES
warnings.filterwarnings('ignore')
SPECS = [('a', a, 'first') for a in ('PF', 'KT', 'KTMD', 'MD', 'SLP')] + [('b', a, 'first') for a in ('PF', 'KT', 'KTMD', 'MD')] \
        + [('b', 'SLP', 'row')]                          # jobs/edges_recompute.pbs, in its order


def windows(banks):
    """Every zns window in file order: (file stem, k, id, bank dir, channel subset)."""
    out = []
    for stage, agent, mode in SPECS:
        d = f'{banks}/neurotycho-long-stage-{stage}'; man = json.load(open(f'{d}/manifest.json'))
        rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
        S0 = sub32(rows[0]['M'])
        out += [(f'zns_{stage.upper()}_{agent}', k, r['id'], d, S0 if mode == 'first' else sub32(r['M'])) for k, r in enumerate(rows)]
    return out


def theiler(v):
    x = (v - v.mean(0)) / v.std(0); T, M = x.shape; A = np.array([acf(x[:, m]) for m in range(M)]); iu = np.triu_indices(M, 1)
    return np.array([[min(int(2 * np.dot(A[i], A[j])), T // 10) for j in range(M)] for i in range(M)])[iu]


def draws(args):
    v, k, bs = args; K, W0 = [], None
    for b in bs:
        s = multivariate_iaaft(v, np.random.default_rng(k if b == 0 else [k, b]), final='spectrum')[0]
        K.append(edge_mi(s)[1])
        if b == 0: W0 = theiler(s)
    return np.array(K, np.float32), W0


def load(w):
    _, k, wid, d, sub = w
    return ns(np.load(f'{d}/windows.npz')[wid].astype(float)[:, sub])


if __name__ == '__main__':
    mode, banks = sys.argv[1], sys.argv[2]; ws = windows(banks)
    if mode == 'gate':
        zdir, n, nw = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        sel = [w for w in ws if w[0] == 'zns_A_PF'][:n]
        with Pool(nw) as p: res = p.map(draws, [(load(w), w[1], [0]) for w in sel], chunksize=1)
        Z = np.load(f'{zdir}/zns_A_PF.npz'); assert list(Z['id'][:n]) == [w[2] for w in sel]
        dev = max(float(np.abs(r[0][0] - Z['E'][w[1], 1, 1, 1]).max()) for r, w in zip(res, sel))
        print(f'gate: max |K draw0 - zns K_surr| over {n} windows = {dev:.3g}', 'PASS' if dev < 1e-4 else 'FAIL', flush=True)
        sys.exit(0 if dev < 1e-4 else 3)
    out = sys.argv[3]
    if mode == 'tier1':
        i, n, B, nw = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        sel = [w for g, w in enumerate(ws) if g % n == i]; blocks = np.array_split(np.arange(B), 4)
        jobs = [(q, list(bl)) for q in range(len(sel)) for bl in blocks]
        X = {q: load(w) for q, w in enumerate(sel)}
        with Pool(nw) as p: res = p.map(draws, [(X[q], sel[q][1], bl) for q, bl in jobs], chunksize=1)
        K = np.empty((len(sel), B, 496), np.float32); W0 = np.empty((len(sel), 496), np.int16)
        for (q, bl), (k_, w0) in zip(jobs, res):
            K[q, bl] = k_
            if 0 in bl: W0[q] = w0
        Wr = np.array([theiler(X[q]) for q in range(len(sel))], np.int16)
    else:                                                # tier2: one window, many draws in blocks
        wid, B, blk, nw = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        sel = [w for w in ws if w[2] == wid]; assert len(sel) == 1, wid; v = load(sel[0])
        blocks = [list(range(a, min(a + blk, B))) for a in range(0, B, blk)]
        with Pool(nw) as p: res = p.map(draws, [(v, sel[0][1], bl) for bl in blocks], chunksize=1)
        K = np.concatenate([r[0] for r in res])[None]; W0 = res[0][1][None].astype(np.int16); Wr = theiler(v)[None].astype(np.int16)
    np.savez(out, K=K, W_rec=Wr, W_s0=W0, zfile=np.array([w[0] for w in sel]), k=np.array([w[1] for w in sel]),
             id=np.array([w[2] for w in sel]))
    print('done', mode, out, K.shape, flush=True)
