"""Same calibration with each channel normal-scored first (rank -> Gaussian quantile), so marginal shape is removed
and only copula (dependence) non-Gaussianity can remain."""
import numpy as np
from scipy.stats import norm, rankdata
from joblib import Parallel, delayed
import mech
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
KINDS = ['linear', 'marginals', 'curved', 'switch', 'local', 'gain']
def run(kind, s):
    rng = np.random.default_rng(s); x, y = mech.synth(kind, 1000, rng); x, y = ns(x), ns(y)
    d, sd = mech.zscore_vs_surr(x, y, rng); return d, sd, np.corrcoef(x, y)[0, 1]
print('normal-scored | (real − IAAFT) / surrogate SD, 16 windows;  r  |  ' + '  '.join(f'{n:>7s}' for n in mech.NAMES))
for k in KINDS:
    R = Parallel(n_jobs=6)(delayed(run)(k, s) for s in range(16))
    D = np.array([q[0] for q in R]); SD = np.array([q[1] for q in R])
    print(f'{k:10s} {np.mean([q[2] for q in R]):.2f} | ' + '  '.join(f'{v:+7.1f}' for v in D.mean(0) / SD.mean(0)))
