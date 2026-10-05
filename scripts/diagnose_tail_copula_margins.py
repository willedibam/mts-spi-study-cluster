"""Focused-probe double dissociation of copula versus marginal alignment; not p90.

Three arms share M=N=16, T=1000, eight independent bivariate modules, rho={.1,.2},
nu={3,30} and the five assignment-swap levels of scripts/scout_tail_alignment.py:
  t_copula_t_margins         original Student-t modules;
  t_copula_gaussian_margins  same t copula, analytic PIT to N(0,1) before standardization;
  gaussian_copula_t_margins  Gaussian copula with rho, analytic PIT to t_nu margins.
Every record is an independent realization (no seed sharing across levels or arms).

Declared before any outcome, as hypotheses rather than requirements:
  robust covariance-squared mean reads rho x marginal-scale alignment, so it should track
  the swap level in arms with t margins and weaken under Gaussian margins (joint copula
  geometry may still move MCD, so chance is not assumed);
  z(Kendall,MI) reads rho x copula-term alignment, so it should track with a t copula and
  not with a Gaussian copula, whose population z(Kendall,MI) is constant.
The swap level is construction truth. With a t copula it is collinear with the mean
upper-tail dependence; the Gaussian-copula arm has zero asymptotic tail dependence and its
sweep is never scored against, or described as, a tail transition.
32 seeds per level (16 fit, 16 exploratory validation) are used instead of 8 because
Spearman on 20 validation records cannot resolve |rho| below about .44. No parameter search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from math import factorial
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermeval
from scipy.integrate import quad
from scipy.special import gammaln,digamma,ndtr,ndtri,stdtr,stdtrit
from scipy.stats import chi2,kendalltau,norm,rankdata,spearmanr,t as student_t
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/order-parameter-inference/dependence-transition-261005/tail-copula-diagnostic'
M,T=16,1000
ARMS=('t_copula_t_margins','t_copula_gaussian_margins','gaussian_copula_t_margins')
SEED,ROBUST_SEED,BOOTSTRAP_SEED=261071,261072,261073
SEEDS,FIT_SEEDS=32,16
PROBES=('Pearson','Spearman','Kendall','KSG_MI')
PAIRS=[(a,b) for a in range(4) for b in range(a+1,4)]
ORDERS=(1,3,5,7,9)


def parameters(swaps):
    nu=np.repeat([3.,30.],4);rho=np.repeat([.1,.2],4)
    for j in range(swaps):rho[j],rho[j+4]=rho[j+4],rho[j]
    return rho,nu


def entropy(d,nu):
    return (gammaln(nu/2)-gammaln((nu+d)/2)+d/2*np.log(nu*np.pi)
            +(nu+d)/2*(digamma((nu+d)/2)-digamma(nu/2)))


def to_gaussian(x,nu):
    """Exact N(0,1) scores of standard t_nu values, evaluated through the lower tail."""
    return -np.sign(x)*ndtri(stdtr(nu,-np.abs(x)))


def to_student(z,nu):
    """Exact standard t_nu quantiles of N(0,1) values, evaluated through the lower tail."""
    return -np.sign(z)*stdtrit(nu,ndtr(-np.abs(z)))


def _coefficient(f,k):
    weight=np.zeros(k+1);weight[k]=1
    value=quad(lambda z:f(z)*hermeval(z,weight)*norm.pdf(z),0,12,limit=200)[0]
    return 2*value/np.sqrt(factorial(k))


@lru_cache(maxsize=None)
def _squared_coefficients(arm,nu):
    """E_W[a_k^2] of the odd Hermite coefficients that carry rho**k (Mehler expansion)."""
    if arm==ARMS[2]:
        return tuple(_coefficient(lambda z:to_student(z,nu),k)**2 for k in ORDERS)
    def mixed(u,k):
        s=np.sqrt(nu/chi2.ppf(u,nu))
        return _coefficient(lambda z:to_gaussian(s*z,nu),k)**2
    return tuple(quad(mixed,0,1,args=(k,),limit=200)[0] for k in ORDERS)


def population_pearson(arm,rho,nu):
    """Population Pearson correlation of one module; exact for the original arm."""
    if arm==ARMS[0]:return float(rho)
    variance=1. if arm==ARMS[1] else nu/(nu-2)
    return float(sum(c*rho**k for c,k in zip(_squared_coefficients(arm,float(nu)),ORDERS))/variance)


def population(arm,swaps):
    rho,nu=parameters(swaps);gaussian=arm==ARMS[2]
    mi=-.5*np.log1p(-rho*rho)+(0 if gaussian else 2*entropy(1,nu)-entropy(2,nu))
    tau=2/np.pi*np.arcsin(rho)
    r=np.array([population_pearson(arm,a,b) for a,b in zip(rho,nu)])
    tail=2*student_t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),nu+1)
    edge=lambda v:np.r_[v,np.zeros(112)]
    return dict(arm=arm,swaps=swaps,mean_r=r.sum()/120,mean_tau=tau.sum()/120,mean_MI=mi.sum()/120,
                Q_tail=np.nan if gaussian else tail.sum()/120,
                z_Kendall_MI=np.corrcoef(edge(tau),edge(mi))[0,1],z_Pearson_MI=np.corrcoef(edge(r),edge(mi))[0,1])


def latent(arm,swaps,seed):
    """Unstandardized module-ordered channels: even rows first members, odd rows partners."""
    rng=np.random.default_rng(np.random.SeedSequence([SEED,ARMS.index(arm),swaps,seed]))
    rho,nu=parameters(swaps);noise=rng.normal(size=(8,2,T))
    first=noise[:,0];second=rho[:,None]*noise[:,0]+np.sqrt(1-rho[:,None]**2)*noise[:,1]
    x=np.empty((M,T))
    if arm==ARMS[2]:
        x[::2]=to_student(first,nu[:,None]);x[1::2]=to_student(second,nu[:,None])
    else:
        scale=np.sqrt(rng.chisquare(nu[:,None],size=(8,T))/nu[:,None])
        x[::2]=first/scale;x[1::2]=second/scale
        if arm==ARMS[1]:x=to_gaussian(x,np.repeat(nu,2)[:,None])
    return x,rng


def recording(arm,swaps,seed):
    x,rng=latent(arm,swaps,seed);x=x[rng.permutation(M)]
    return (x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)


def estimate(task):
    arm,swaps,seed=task;x=recording(arm,swaps,seed);ii,jj=np.triu_indices(M,1)
    pearson=np.corrcoef(x)[ii,jj];spearman=np.corrcoef(rankdata(x,axis=1))[ii,jj]
    kendall=np.array([kendalltau(x[i],x[j]).statistic for i,j in zip(ii,jj)])
    mi=np.zeros((M,M))
    for i in range(M-1):
        mi[i,i+1:]=mutual_info_regression(x[i+1:].T,x[i],n_neighbors=4,random_state=ROBUST_SEED)
    covariance=EllipticEnvelope(random_state=ROBUST_SEED).fit(x.T).covariance_
    variance=np.diag(covariance);robust=covariance[ii,jj]
    profiles=np.array([pearson,spearman,kendall,mi[ii,jj]])
    z=np.corrcoef(profiles)[tuple(zip(*PAIRS))]
    row=dict(arm=arm,swaps=swaps,seed=seed,role='fit' if seed<FIT_SEEDS else 'validation',
             **{'mean_'+n:v for n,v in zip(PROBES,profiles.mean(1))},
             robust_cov_sq_mean=np.mean(robust**2),
             robust_cor_sq_mean=np.mean(robust**2/(variance[ii]*variance[jj])),
             mean_robust_variance=variance.mean(),min_robust_variance=variance.min(),
             z_Kendall_MI=z[PAIRS.index((2,3))],z_Pearson_MI=z[PAIRS.index((0,3))],
             z_robustcov_Kendall=np.corrcoef(robust,kendall)[0,1])
    return row,z


def coordinate(x,fit,standard):
    """Target-blind PC1 fitted on fit rows only; mirrors report_dependence_transition.coordinate."""
    center=x[fit].mean(0);scale=x[fit].std(0) if standard else 1.
    x=(x-center)/scale;model=PCA(n_components=1,svd_solver='full').fit(x[fit])
    return model.transform(x)[:,0],float(model.explained_variance_ratio_[0])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(rows,z):
    rng=np.random.default_rng(BOOTSTRAP_SEED);metrics=[];curves=[];scores=rows.copy()
    features=['mean_'+n for n in PROBES]+['robust_cov_sq_mean','robust_cor_sq_mean','mean_robust_variance',
              'z_Kendall_MI','z_Pearson_MI','z_robustcov_Kendall']
    for arm in ARMS:
        part=(rows.arm==arm).to_numpy();fit=(rows.role=='fit').to_numpy()[part]
        means=rows.loc[part,['mean_'+n for n in PROBES]].to_numpy()
        evr={}
        for label,x,standard in [('mean_PC1',means,True),('z_PC1',z[part],False),('z_standard_PC1',z[part],True)]:
            scores.loc[part,label],evr[label]=coordinate(x,fit,standard)
        level=rows.loc[part,'swaps'].to_numpy();held=np.flatnonzero(~fit)
        draws=rng.integers(0,len(held),size=(2000,len(held)))
        for label in features+list(evr):
            q=scores.loc[part,label].to_numpy()
            rho=spearmanr(q[held],level[held]).statistic
            boot=[spearmanr(q[held[d]],level[held[d]]).statistic for d in draws]
            if label in evr:
                # PC sign is arbitrary: orient by fit rows only, then report the signed validation value.
                sign=1. if spearmanr(q[fit],level[fit]).statistic>=0 else -1.
                rho*=sign;boot=sign*np.array(boot)
            low,high=np.nanpercentile(boot,[2.5,97.5])
            metrics.append(dict(arm=arm,method=label,validation_rho_swap_level=rho,ci_low=low,ci_high=high,
                                excludes_zero=bool(low>0 or high<0),n=len(held),fit_rho=spearmanr(q[fit],level[fit]).statistic,
                                evr=evr.get(label,np.nan)))
            for s in range(5):
                v=q[held][level[held]==s]
                curves.append(dict(arm=arm,method=label,swaps=s,validation_mean=v.mean(),validation_sd=v.std(ddof=1)))
    return scores,pd.DataFrame(metrics),pd.DataFrame(curves)


def run(out,workers):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'rows.csv').exists():raise FileExistsError(out/'rows.csv')
    pop=pd.DataFrame([population(arm,j) for arm in ARMS for j in range(5)])
    for arm,part in pop.groupby('arm'):
        assert np.ptp(part[['mean_tau','mean_MI']].to_numpy(),axis=0).max()<1e-14
    assert np.ptp(pop.loc[pop.arm==ARMS[0],'mean_r'])<1e-14
    assert np.ptp(pop.loc[pop.arm==ARMS[2],'z_Kendall_MI'])<1e-12
    pop.to_csv(out/'population.csv',index=False)
    tasks=[(arm,j,seed) for arm in ARMS for j in range(5) for seed in range(SEEDS)]
    rows=[];z=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(row,v) in enumerate(pool.map(estimate,tasks,chunksize=4)):
            rows.append(row);z.append(v)
            if i%48==0:print('copula/margin scout',i+1,'/',len(tasks),flush=True)
    rows=pd.DataFrame(rows);z=np.array(z)
    scores,metrics,curves=analyze(rows,z)
    scores.to_csv(out/'rows.csv',index=False);metrics.to_csv(out/'metrics.csv',index=False)
    curves.to_csv(out/'curves.csv',index=False)
    np.savez_compressed(out/'features.npz',z=z,pairs=np.array([f'{PROBES[a]}-{PROBES[b]}' for a,b in PAIRS]))
    (out/'report.json').write_text(json.dumps(dict(records=len(rows),M=M,T=T,arms=ARMS,source_sha256=sha(__file__),
        seeds=dict(recording=f'SeedSequence([{SEED},arm_index,swaps,seed])',robust_and_ksg=ROBUST_SEED,bootstrap=BOOTSTRAP_SEED),
        split=f'seeds0-{FIT_SEEDS-1} fit PC1/orientation, seeds{FIT_SEEDS}-{SEEDS-1} exploratory validation',
        scope='Focused probes only (Pearson, Spearman, Kendall, KSG MI, sklearn EllipticEnvelope), not p90 and not confirmation. '
              'Independent realization per arm/level/seed. Target is the assignment-swap level (construction truth); it is collinear '
              'with mean upper-tail dependence only for the two t-copula arms. The Gaussian-copula arm has zero asymptotic tail '
              'dependence and is not a tail transition. Population Pearson is exactly matched only with t copula and t margins.',
        intervals='Percentile bootstrap over validation records, 2000 draws; exploratory and unadjusted.'),indent=2)+'\n')
    table=metrics.pivot(index='method',columns='arm',values='validation_rho_swap_level')[list(ARMS)]
    print(pop.round(6).to_string(index=False));print(table.round(3).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.output,a.workers)
