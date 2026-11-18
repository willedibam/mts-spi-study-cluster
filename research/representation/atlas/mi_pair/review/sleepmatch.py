"""Delta-matched drug vs sleep contrast of the z(Gaussian MI, kNN MI) residual beyond IAAFT (10-s, stages A+B, M=32)."""
import numpy as np
L = [np.load(f) for f in ('decide10_nt.npz', 'decide10b_nt.npz')]
cat = lambda k: np.concatenate([d[k] for d in L])
i = list(L[0]['names']).index('zgk')
res = np.concatenate([d['V'][:, 0, i] - d['V'][:, 1, i] for d in L])
unit, state, agent, animal, df = cat('unit'), cat('state'), cat('agent'), cat('animal'), cat('delta_fraction').astype(float)
d = res.copy()
for u in np.unique(unit):
    m = unit == u; d[m] -= res[m & (state == 'awake_ec')].mean()
eo = state == 'awake_eo'; sdw = d[eo].std(ddof=1)            # window-level null scale
grp = np.where(state == 'drug', agent, np.where(state == 'sleep', 'SLP', state))
print(f'window null SD (EO-EC) {sdw:.4f}; all in SD units below\n')
bins = [.15, .4, .55, .65, .75, .85, .97]
print('delta bin        ' + ''.join(f'{bins[k]:.2f}-{bins[k+1]:.2f}   ' for k in range(len(bins) - 1)))
for an in ('George', 'Chibi'):
    for g in ('awake_ec', 'awake_eo', 'SLP', 'KT', 'KTMD', 'MD', 'PF'):
        m = (animal == an) & (grp == g)
        if not m.any(): continue
        cells = []
        for k in range(len(bins) - 1):
            s = m & (df >= bins[k]) & (df < bins[k + 1])
            cells.append(f'{d[s].mean()/sdw:+5.1f} ({s.sum():3d})  ' if s.sum() >= 3 else '    .        ')
        print(f'{an:6s} {g:9s} ' + ''.join(cells))
    print()
# matched overlap region: session means
lo, hi = .65, .88
print(f'matched region delta in [{lo},{hi}): session means (SD), mean delta')
for g in ('awake_ec', 'SLP', 'KT', 'KTMD', 'MD', 'PF'):
    out = []
    for u in np.unique(unit[grp == g]):
        s = (unit == u) & (grp == g) & (df >= lo) & (df < hi)
        if s.sum() >= 3: out.append((d[s].mean() / sdw, df[s].mean(), s.sum(), animal[s][0][0]))
    if out:
        a = np.array([o[0] for o in out])
        print(f'{g:9s} n_sess={len(a):2d} mean {a.mean():+.2f}  down {np.sum(a<0)}/{len(a)}  delta {np.mean([o[1] for o in out]):.3f}  ' + ' '.join(f'{o[3]}{o[0]:+.1f}' for o in out))
# within-animal regression d ~ animal + beta*delta + state effects (ec,eo baseline), session bootstrap
X_names = ['SLP', 'KT', 'KTMD', 'MD', 'PF']
use = np.isin(grp, ['awake_ec', 'awake_eo'] + X_names)
def fit(idx):
    X = np.c_[np.ones(len(idx)), (animal[idx] == 'George'), df[idx]] 
    X = np.c_[X, *[(grp[idx] == g) for g in X_names]].astype(float)
    return np.linalg.lstsq(X, d[idx] / sdw, rcond=None)[0][2:]
idx = np.where(use)[0]; est = fit(idx)
us = np.unique(unit[use]); rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    pick = rng.choice(us, len(us)); bs.append(fit(np.concatenate([idx[unit[idx] == u] for u in pick])))
bs = np.array(bs); lo_, hi_ = np.percentile(bs, [2.5, 97.5], 0)
print('\nregression (SD units; session bootstrap 95% CI): per +0.1 delta, and state effect at equal delta')
for n, e, a, b in zip(['delta x0.1'] + X_names, est, lo_, hi_):
    s = .1 if n.startswith('delta') else 1
    print(f'  {n:10s} {e*s:+.2f}  [{a*s:+.2f}, {b*s:+.2f}]')
# slopes within state
print('\nwithin-state slope per +0.1 delta (SD), within animal:')
for g in ('awake_ec', 'awake_eo', 'SLP', 'drug'):
    m = (grp == g) if g != 'drug' else (state == 'drug')
    for an in ('George', 'Chibi'):
        s = m & (animal == an)
        if s.sum() > 10:
            # demean per unit to get within-session slope
            dd, xx = d[s].copy(), df[s].copy()
            for u in np.unique(unit[s]): q = unit[s] == u; dd[q] -= dd[q].mean(); xx[q] -= xx[q].mean()
            print(f'  {g:9s} {an:6s} {np.polyfit(xx, dd/sdw, 1)[0]*.1:+.2f}  (n={s.sum()})')
