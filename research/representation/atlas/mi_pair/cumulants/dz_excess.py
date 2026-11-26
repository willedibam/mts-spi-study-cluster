"""Post hoc (26 November), descriptive: what moves Δz = corr(G, K)_real - corr(G, K)_surrogate, and does the per-pair MI excess
track the pair's fourth-order cumulants? Per window, from the zns edge arrays (normal-scored, 496 pairs) and ke_*.npz:
  zr, zs      Fisher z of corr(G, K), recording and its single IAAFT draw (panel (c) differences the untransformed r);
  sdG, mG     spread and mean over pairs of Gaussian MI G; sdE, mE: of the excess E = K - G; sdEs: the surrogate's sdE;
  absk04      mean |k04| over pairs (k04 signed mean cancels when pairs are kurtotic in both directions); negfrac: share k04 < 0;
  rmsz04      root mean square over pairs of k04's z against the 100 marginal-matched IAAFT draws ('im');
  rho_E_*     within-window Spearman over pairs of E with k04, |k04| and Mardia M (all scale with |r|);
  rho_X_absz  the same for X = E - E_surrogate (excess beyond the IAAFT draw) with |k04 z against 'im'|: both sides null-controlled.
  python dz_excess.py <zns dir> <ensemble dir>
"""
import glob, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def main(zd, ed):
    Z = {}
    for f in sorted(glob.glob(f'{zd}/zns_[AB]_*.npz')):
        d = np.load(f); Z.update(zip(d['id'], d['E'][:, 1].astype(float)))
    R = []
    for f in sorted(glob.glob(f'{ed}/ke_[AB]_*.npz')):
        d = np.load(f)
        for i, w in enumerate(d['id']):
            (g, k), (gs, ks) = Z[w]; e = k - g
            re, mu, sd = (d[a][i].astype(float) for a in ('real', 'im_mean', 'im_sd'))
            assert np.allclose(-.5 * np.log(1 - re[0] ** 2), g, atol=1e-4)        # same pairs, same order
            k04, M = re[3], re[4]; z04 = (k04 - mu[3]) / sd[3]
            R.append(dict(animal=d['animal'][i], grp=d['agent'][i] if d['state'][i] == 'drug' else d['state'][i],
                          zr=np.arctanh(np.corrcoef(g, k)[0, 1]), zs=np.arctanh(np.corrcoef(gs, ks)[0, 1]),
                          mG=g.mean(), sdG=g.std(), mE=e.mean(), sdE=e.std(), mEs=(ks - gs).mean(), sdEs=(ks - gs).std(),
                          k04=k04.mean(), absk04=np.abs(k04).mean(), negfrac=(k04 < 0).mean(), rmsz04=np.sqrt((z04 ** 2).mean()),
                          rho_E_k04=spearmanr(e, k04)[0], rho_E_absk04=spearmanr(e, np.abs(k04))[0], rho_E_M=spearmanr(e, M)[0],
                          rho_X_absz=spearmanr(e - (ks - gs), np.abs(z04))[0],
                          delta=d['delta_fraction'][i]))
    pd.set_option('display.width', 250)
    print(pd.DataFrame(R).groupby(['animal', 'grp']).mean().round(3).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:3])
