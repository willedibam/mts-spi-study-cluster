"""Post hoc (matched-result.md): does within-window amplitude non-stationarity drive the macaque residuals? Envelope CV over 8 blocks of
25 samples and the mean correlation of channel block envelopes, real minus the joint-IAAFT surrogate; window-level association with the
MI, Chebyshev and CCE nets; the drug contrast adjusted by a within-state slope; breadth of all usable coordinates against a split-half
control. Run from the repo root: python matched_drivers.py <dir with atlas/discovery, nt-ns/windows.npz, nt-iaaft/windows.npz>"""
import sys, json, numpy as np
sys.path[:0] = ['.', 'src', 'research/representation/atlas/movers']
import matched as M
from scipy.signal import hilbert
from scipy.stats import ttest_1samp, pearsonr
S = sys.argv[1]; A = S + '/atlas'
R, Sg, spis, rows = M._load(A, 'nt-ns', 'nt-iaaft'); zr, zs = R['z'], Sg['z']
a, b, co, mins, cls = M._minimal(spis)
ids = [str(x) for x in R['ids']]
unit = np.array([r['unit'] for r in rows]); st = np.array([r['state'] for r in rows]); t0 = np.array([r['start_s'] for r in rows])
agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
prim = sorted(u for u in set(unit) if agent[u] in M.PRIMARY and all(((unit == u) & (st == s)).any() for s in M.STATES))
G = lambda u, s: np.flatnonzero((unit == u) & (st == s))
# within-window amplitude non-stationarity on a bank: per channel block RMS of the Hilbert envelope (8 blocks of 25 samples)
def feats(bank):
    W = np.load(f'{S}/{bank}/windows.npz'); out = np.zeros((len(ids), 3))
    for n, k in enumerate(ids):
        x = W[k]; x = (x - x.mean(0)) / x.std(0); env = np.abs(hilbert(x, axis=0))
        blk = env[:200].reshape(8, 25, -1).mean(1)                      # 8 x M
        cv = (blk.std(0) / blk.mean(0)).mean()                          # local amplitude non-stationarity
        bz = (blk - blk.mean(0)) / (blk.std(0) + 1e-12); C = np.corrcoef(bz.T); iu = np.triu_indices_from(C, 1)
        shared = np.nanmean(C[iu])                                      # shared gain: co-fluctuation of block envelopes
        kurt = np.mean(((x - x.mean(0)) / x.std(0))[:, :] ** 4) - 3     # zero-ish after normal-scoring; check
        out[n] = cv, shared, kurt
    return out
Fr, Fs = feats('nt-ns'), feats('nt-iaaft')
names = ['envelope CV', 'shared envelope r', 'kurtosis']
cA = co('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'); cC = co('pdist_chebyshev', 'pdist_canberra'); cE = co('cce_gaussian_n-5', 'cce_kozachenko_n-5')
netw = lambda c: zr[:, c] - zs[:, c]
print('state means over primary sessions (real, surrogate):')
for s in M.STATES:
    ix = np.concatenate([G(u, s) for u in prim]); print(f'  {s:9s}', ' '.join(f'{nm} {Fr[ix, i].mean():.3f}/{Fs[ix, i].mean():.3f}' for i, nm in enumerate(names)))
# session-level: does Delta(real - surr feature) predict Delta net?
dF = np.array([[ (Fr[G(u,'drug'), i] - Fs[G(u,'drug'), i]).mean() - (Fr[G(u,M.REF), i] - Fs[G(u,M.REF), i]).mean() for i in range(2)] for u in prim])
for c, nm in ((cA, 'MI'), (cC, 'Chebyshev'), (cE, 'CCE')):
    dn = np.array([np.nanmean(netw(c)[G(u, 'drug')]) - np.nanmean(netw(c)[G(u, M.REF)]) for u in prim])
    print(f'{nm}: session Delta net {dn.mean():+.4f}; r with Delta envelope-CV excess {pearsonr(dF[:,0], dn)[0]:+.2f}, with Delta shared-envelope excess {pearsonr(dF[:,1], dn)[0]:+.2f}')
    # within-state window level, session-demeaned, drug windows and awake windows
    for s in ('drug', M.REF):
        xs, ys, zs_ = [], [], []
        for u in prim:
            ix = G(u, s); y = netw(c)[ix]; ok = np.isfinite(y)
            xs += list((Fr[ix, 0] - Fs[ix, 0])[ok] - (Fr[ix, 0] - Fs[ix, 0])[ok].mean()); zs_ += list((Fr[ix, 1] - Fs[ix, 1])[ok] - (Fr[ix, 1] - Fs[ix, 1])[ok].mean()); ys += list(y[ok] - y[ok].mean())
        print(f'   window-level within {s}: r(net, envelope-CV excess) {pearsonr(xs, ys)[0]:+.2f}, r(net, shared excess) {pearsonr(zs_, ys)[0]:+.2f}, n {len(ys)}')
# do MI and Chebyshev nets co-vary across windows (common driver)?
for s in ('drug', M.REF):
    xa, xc = [], []
    for u in prim:
        ix = G(u, s); p, q = netw(cA)[ix], netw(cC)[ix]; ok = np.isfinite(p) & np.isfinite(q)
        xa += list(p[ok] - p[ok].mean()); xc += list(q[ok] - q[ok].mean())
    print(f'r(MI net, Chebyshev net) within {s}: {pearsonr(xa, xc)[0]:+.2f}')
# breadth: all usable coordinates, drug against within-awake split-half control
fin = np.ones(len(a), bool)
for u in prim:
    for s in M.STATES:
        for Z in (zr, zs): fin &= np.isfinite(Z[G(u, s)]).mean(0) >= M.FIN
z0 = np.mean([np.nanmean(zr[G(u, M.REF)], 0) for u in prim], 0); rel, relok = M._rel(A, 'nt-ns', spis, a, b)
usable = fin & (np.abs(z0) < M.CEIL) & relok
net = lambda ix: np.nanmean(zr[ix], 0) - np.nanmean(zs[ix], 0)
def halves(u):
    ix = G(u, M.REF); ix = ix[np.argsort(t0[ix])]; return ix[:5], ix[5:]
an = np.array([animal[u] for u in prim]); animals = sorted(set(an))
def count(X):
    c = np.flatnonzero(usable); x = X[:, c]; p = ttest_1samp(x, 0).pvalue; q = M.bh(p)
    every = np.all([np.sign(x[an == k].mean(0)) == np.sign(x.mean(0)) for k in animals], 0); return int(((q <= .05) & every).sum()), len(c)
H = np.array([net(halves(u)[1]) - net(halves(u)[0]) for u in prim])
D5 = np.array([net(G(u, 'drug')[:5]) - net(halves(u)[0]) for u in prim])   # drug with 5 windows against the same 5 awake: power-matched to the split-half
print('all usable: split-half control', count(H), '; power-matched drug (5 drug windows vs first awake half)', count(D5))
# drug effect adjusted for the window's envelope-CV excess (session fixed effects; awake_ec and drug windows)
print('\nadjusted drug effect (session fixed effects), per-session estimate then t over 19 sessions and animal means:')
for c, nm in ((cA, 'MI'), (cC, 'Chebyshev'), (cE, 'CCE')):
    raw, adj, slope = [], [], []
    # common slope estimated from within-state variation only (demeaned per session x state), so it cannot absorb the drug contrast
    xs, ys = [], []
    for u in prim:
        for s in ('drug', M.REF):
            ix = G(u, s); y = netw(c)[ix]; x = Fr[ix, 0] - Fs[ix, 0]; ok = np.isfinite(y); xs += list(x[ok] - x[ok].mean()); ys += list(y[ok] - y[ok].mean())
    bta = np.polyfit(xs, ys, 1)[0]
    for u in prim:
        d, w = G(u, 'drug'), G(u, M.REF); yd, yw = netw(c)[d], netw(c)[w]; xd, xw = (Fr[d, 0] - Fs[d, 0]), (Fr[w, 0] - Fs[w, 0])
        raw.append(np.nanmean(yd) - np.nanmean(yw)); adj.append(np.nanmean(yd) - np.nanmean(yw) - bta * (xd.mean() - xw.mean()))
    raw, adj = np.array(raw), np.array(adj); am = lambda v: [v[an == k].mean() for k in animals]
    print(f'  {nm}: within-state slope {bta:+.3f}; raw {raw.mean():+.4f} ({(raw<0).sum()}/19, animal t {ttest_1samp(am(raw),0).statistic:+.1f}); '
          f'adjusted {adj.mean():+.4f} ({(adj<0).sum()}/19, animal t {ttest_1samp(am(adj),0).statistic:+.1f}); share removed {1-adj.mean()/raw.mean():.2f}')
