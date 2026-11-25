"""Autocorrelation of the macaque 10-s windows as the MI-pair estimator sees them (preregs/robust-prereg.md, descriptive).

Normal-scored windows, the edges.py 32-channel subsets (`first` per agent and stage; stage-B SLP `row`), ACF by edges.acf.
Per window and channel: tau_1/e (first 1/e crossing, linear interpolation), envelope tau (1/e crossing of the upper envelope max_{s>=lag} |acf(s)|),
first zero, integral to the first zero, minimum within 1 s, a secondary-peak flag, the e-folding time fitted to log acf
over lags up to tau_1/e, and acf at 2 tau_1/e. Per window and pair: the uncapped AUTO window 2 <a_i, a_j>.

  python robust_acf.py <stage-a bank> <stage-b bank> <out.npz>       compute
  python robust_acf.py summary <out.npz>                             text summary
  python robust_acf.py figure <out.npz> <figure stem>                figure
"""
import sys, json, numpy as np
from edges import acf, ns, sub32, STATES
FS, LMAX = 100, 300  # Hz; lags kept (3 s)
ORDER = ('awake_ec', 'awake_eo', 'KT', 'KTMD', 'MD', 'PF', 'sleep', 'recovery_ec')


def cross(a, lev):
    b = np.nonzero(a < lev)[0]
    if not len(b): return np.nan
    i = b[0]; return i - 1 + (a[i - 1] - lev) / (a[i - 1] - a[i])


def chan_stats(a):
    t1 = cross(a, 1 / np.e); env = np.maximum.accumulate(np.abs(a)[::-1])[::-1]  # upper envelope: max |acf| at this lag or later
    z0 = cross(a, 0); iz = np.sum(a[:int(z0) + 1]) if np.isfinite(z0) else np.nan
    m = a[:FS + 1].min(); d = np.diff(a[:FS + 1]); mx = (d[:-1] > 0) & (d[1:] <= 0); mn = (d[:-1] < 0) & (d[1:] >= 0)
    pk = np.any(mx & (a[1:FS] > .1) & (np.cumsum(mn) > 0))  # secondary peak above .1 after a trough, within 1 s
    L = np.arange(1, int(np.floor(t1)) + 1) if np.isfinite(t1) and t1 >= 1 else np.arange(1, 2)
    fit = -np.sum(L ** 2) / np.sum(L * np.log(np.clip(a[L], 1e-6, None)))  # least squares of log a = -lag / tau through 0
    a2 = np.interp(2 * t1, np.arange(len(a)), a) if np.isfinite(t1) else np.nan
    return [t1, cross(env, 1 / np.e), z0, iz, m, pk, fit, a2]


def compute(da, db, out):
    R, ACF, WA = [], [], []
    for d, stage in ((da, 'A'), (db, 'B')):
        man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
        for ag in ('PF', 'KT', 'KTMD', 'MD', 'SLP'):
            rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] == ag]; S0 = sub32(rows[0]['M'])
            for r in rows:
                x = ns(Wn[r['id']].astype(float)[:, S0 if (stage == 'A' or ag != 'SLP') else sub32(r['M'])])
                x = (x - x.mean(0)) / x.std(0); A = np.array([acf(x[:, m]) for m in range(x.shape[1])])
                ACF.append(A[:, :LMAX + 1].astype(np.float32)); WA.append((2 * A @ A.T)[np.triu_indices(32, 1)].astype(np.float32))
                R.append(dict(stat=np.array([chan_stats(a) for a in A]), unit=r['unit'], state=r['state'], agent=ag, animal=r['animal']))
    np.savez(out, ACF=np.array(ACF), WA=np.array(WA), S=np.array([q['stat'] for q in R]),
             **{k: np.array([q[k] for q in R]) for k in ('unit', 'state', 'agent', 'animal')})


def group(D):
    st, ag = D['state'], D['agent']; return np.where(st == 'drug', ag, st)


def summary(f):
    D = np.load(f); g = group(D); S = D['S']; WA = D['WA']
    names = ('tau_1/e', 'env tau', 'first zero', 'int to zero', 'min <1 s', 'peak', 'fit tau', 'acf(2tau)')
    print('per window x channel, seconds (median [IQR]); fraction oscillatory = min < -.1 within 1 s or secondary peak')
    for s in ORDER:
        m = g == s; q = S[m].reshape(-1, S.shape[-1]); osc = (q[:, 4] < -.1) | (q[:, 5] > 0)
        p = lambda c: f'{np.nanmedian(q[:, c]) / FS:.3f} [{np.nanpercentile(q[:, c], 25) / FS:.3f}, {np.nanpercentile(q[:, c], 75) / FS:.3f}]'
        wa = WA[m].ravel()
        print(f'{s:12s} n_win {m.sum():4d} | tau_1/e {p(0)} | 2tau {2 * np.nanmedian(q[:, 0]) / FS:.3f} | env {p(1)} | fit {p(6)} | '
              f'acf(2tau) {np.nanmedian(q[:, 7]):.3f} | zero {p(2)} | int0 {np.nanmedian(q[:, 3]) / FS:.3f} | osc {osc.mean():.2f} | '
              f'AUTO w {np.median(wa):.1f} [{np.percentile(wa, 25):.1f}, {np.percentile(wa, 75):.1f}] samples, >=100 {(wa >= 100).mean():.3f}, >=20 {(wa >= 20).mean():.2f}')
        # unit x channel averaged ACF tau
    print('\nunit x state x channel averaged ACF: tau_1/e median [IQR] s')
    for s in ORDER:
        m = g == s; taus = []
        for u in np.unique(D['unit'][m]):
            A = D['ACF'][m & (D['unit'] == u)].mean(0); taus += [cross(a, 1 / np.e) / FS for a in A]
        print(f'{s:12s} {np.nanmedian(taus):.3f} [{np.nanpercentile(taus, 25):.3f}, {np.nanpercentile(taus, 75):.3f}]')


if __name__ == '__main__':
    if sys.argv[1] == 'summary': summary(sys.argv[2])
    elif sys.argv[1] == 'figure':
        from robust_acf_fig import figure; figure(sys.argv[2], sys.argv[3])
    else: compute(sys.argv[1], sys.argv[2], sys.argv[3])
