"""A stationary latent-state process with fixed univariate laws and covariance.

Two groups share a background factor. Their additional factors have correlation
alpha*S(t), with S a stationary balanced two-state chain. Each channel is iid
N(0,1) for every alpha; unconditional second-order cross-moments are also fixed.
This is covariance modulation conditional on a latent state, not nonstationarity
of the unconditional process or identified causal coupling.
"""
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans


def simulate(alpha, c, d, family, seed, t=1000):
    d = np.asarray(d, dtype=float)
    if not (0 <= alpha <= 1 and 0 < c and np.all(d > 0) and np.all(c+d < 1)):
        raise ValueError('invalid factor parameters')
    if family not in ('persistent', 'iid') or len(d) % 2:
        raise ValueError('unknown family or uneven full groups')
    rng = np.random.default_rng(seed)
    p = .05 if family == 'persistent' else .5
    s = rng.choice([-1., 1.]) * np.cumprod(np.where(rng.random(t) < p, -1., 1.))
    g, h, j = rng.normal(size=(3, t))
    group = np.arange(len(d)) >= len(d)//2
    factor = np.where(group[None, :], (alpha*s*h + np.sqrt(1-alpha**2)*j)[:, None], h[:, None])
    x = np.sqrt(c)*g[:, None] + np.sqrt(d)*factor + np.sqrt(1-c-d)*rng.normal(size=(t,len(d)))
    return x, s


def population_covariance(c, d, alpha=0., state=0.):
    d=np.asarray(d); group=np.arange(len(d)) >= len(d)//2
    same=group[:,None] == group[None,:]
    r=c+np.sqrt(d[:,None]*d[None,:])*np.where(same,1.,alpha*state)
    np.fill_diagonal(r,1.)
    return r


def summaries(x):
    return np.r_[np.mean(x), np.std(x), np.quantile(x,[.1,.25,.5,.75,.9])]


def raw_references(view):
    """No generator parameters or true group membership enter these features."""
    x=np.asarray(view,dtype=float); x=(x-x.mean(0))/x.std(0)
    t,m=x.shape; iu=np.triu_indices(m,1)
    r=x.T@x/t
    fourth=(x*x).T@(x*x)/t - 1 - 2*r*r
    covariance=summaries(r[iu])
    cumulants=np.r_[summaries(fourth[iu]), summaries(fourth[iu]/np.maximum(1-r[iu]**2,.05))]
    marginal=np.r_[summaries(skew(x,axis=0)),summaries(kurtosis(x,axis=0)),
                   summaries(np.mean(x[:-1]*x[1:],axis=0))]
    windows=[]
    for width in [25,100]:
        local=[]
        for start in range(0,t-width+1,width):
            block=x[start:start+width]
            local.append(np.corrcoef(block.T)[iu])
        windows.extend(summaries(np.var(local,axis=0)))
    # A model-informed moment estimate with groups inferred from observed
    # covariance. Eigenvector sign is immaterial; no true labels are supplied.
    _,v=np.linalg.eigh(r)
    groups=KMeans(2,n_init=10,random_state=0).fit_predict(v[:,-2:])
    between=groups[iu[0]] != groups[iu[1]]
    if between.any() and (~between).any():
        c_hat=np.median(r[iu][between])
        d_hat=max(np.median(r[iu][~between])-c_hat,.025)
        proxy=np.sqrt(max(np.median(fourth[iu][between]),0)/(2*d_hat*d_hat))
    else:
        proxy=0.
    return dict(marginal=marginal,covariance=covariance,cumulant=cumulants,
                window=np.asarray(windows),moment_proxy=np.asarray([np.clip(proxy,0,1)]))
