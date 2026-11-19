"""Post hoc diagnostics of the matched macaque screen (matched-result.md), written after its output was read: which pairs the
eyes-open control finds, an awake eyes-closed split-half control (first against second half of the windows in time), and each
drug discovery's size against both. Run from the repo root: python matched_posthoc.py <atlas dir> <out.npz> > controls.txt"""
import sys, json, numpy as np
from pathlib import Path
sys.path[:0] = ['.', 'src', 'research/representation/atlas/movers']
import matched as M
from scipy.stats import ttest_1samp
A = sys.argv[1]  # atlas dir holding discovery/nt-{ns,iaaft}-reduced.npz and nt-ns-reliability-all.json
R, S, spis, rows = M._load(A, 'nt-ns', 'nt-iaaft'); zr, zs, dr, ds = R['z'], S['z'], R['zdc'], S['zdc']
a, b, co, mins, cls = M._minimal(spis)
unit = np.array([r['unit'] for r in rows]); st = np.array([r['state'] for r in rows]); t0 = np.array([r['start_s'] for r in rows])
agent = {r['unit']: r['agent'] for r in rows}; animal = {r['unit']: r['animal'] for r in rows}
prim = sorted(u for u in set(unit) if agent[u] in M.PRIMARY and all(((unit == u) & (st == s)).any() for s in M.STATES))
G = lambda u, s: np.flatnonzero((unit == u) & (st == s))
m = lambda Z, ix: np.nanmean(Z[ix], 0)
net = lambda ix, P=zr, N=zs: m(P, ix) - m(N, ix)
fin = np.ones(len(a), bool)
for u in prim:
    for s in M.STATES:
        for Z in (zr, zs): fin &= np.isfinite(Z[G(u, s)]).mean(0) >= M.FIN
z0 = np.mean([m(zr, G(u, M.REF)) for u in prim], 0)
rel, relok = M._rel(A, 'nt-ns', spis, a, b)
usable = fin & (np.abs(z0) < M.CEIL) & relok
Dn = np.array([net(G(u, 'drug')) - net(G(u, M.REF)) for u in prim]); Cn = np.array([net(G(u, 'awake_eo')) - net(G(u, M.REF)) for u in prim])
Ddc = np.array([net(G(u, 'drug'), dr, ds) - net(G(u, M.REF), dr, ds) for u in prim])
def halves(u):
    ix = G(u, M.REF); ix = ix[np.argsort(t0[ix])]; h = len(ix) // 2; return ix[:h], ix[h:]
Hn = np.array([net(halves(u)[1]) - net(halves(u)[0]) for u in prim])
an = np.array([animal[u] for u in prim]); animals = sorted(set(an))
def screen(mask, X):
    c = np.flatnonzero(mask); x = X[:, c]; mu = x.mean(0); p = ttest_1samp(x, 0).pvalue; q = M.bh(p)
    every = np.all([np.sign(x[an == k].mean(0)) == np.sign(mu) for k in animals], 0)
    return c, p, q, (q <= M.Q) & every
mu_ = mins & usable
c, p, q, d = screen(mu_, Dn); _, pc, qc, dc_ = screen(mu_, Cn); _, ph, qh, dh = screen(mu_, Hn)
print('primary', len(c), 'drug', d.sum(), 'eyes-open control', dc_.sum(), 'post hoc awake split-half control', dh.sum())
print('overlap drug & eo', (d & dc_).sum())
name = lambda k: f'{spis[a[k]]}|{spis[b[k]]} [{cls.get(k)}]'
print('\n-- eyes-open control discoveries: EO mean, drug mean, drug q')
for i in np.flatnonzero(dc_)[np.argsort(pc[dc_])]:
    k = c[i]; print(f'  {name(k)} eo {Cn[:, k].mean():+.4f} ({(np.sign(Cn[:,k])==np.sign(Cn[:,k].mean())).sum()}/19) drug {Dn[:, k].mean():+.4f} q {q[i]:.2g} disc {d[i]}')
print('\n-- split-half control discoveries')
for i in np.flatnonzero(dh): k = c[i]; print(f'  {name(k)} half {Hn[:, k].mean():+.4f} drug {Dn[:, k].mean():+.4f}')
print('\n-- dc vs eo check for band rows (full precision)')
for i in np.flatnonzero(d):
    k = c[i]
    if 'band' in str(cls.get(k)): print(' ', name(k), repr(Ddc[:, k].mean()), repr(Cn[:, k].mean()))
# drug discoveries: drug effect relative to EO and split-half spread
print('\n-- drug discoveries: |drug| / |eo|, drug sign vs eo sign, drug/SD(half)')
sdh = Hn.std(0, ddof=1)
out = []
for i in np.flatnonzero(d):
    k = c[i]; dm, em = Dn[:, k].mean(), Cn[:, k].mean()
    out.append((name(k), dm, em, abs(dm) / max(abs(em), 1e-9), dm / sdh[k], q[i], qc[i], Ddc[:, k].mean()))
for r in sorted(out, key=lambda r: -r[3]): print(f'  {r[0]} drug {r[1]:+.4f} eo {r[2]:+.4f} ratio {r[3]:.1f} drug/SDhalf {r[4]:+.1f} q {r[5]:.2g} qeo {r[6]:.2g} dc {r[7]:+.4f}')
np.savez(Path(sys.argv[2]), Hn=Hn, Cn=Cn, Dn=Dn, Ddc=Ddc, usable=usable, mins=mins)
