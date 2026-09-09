"""Common-drive oscillations with aligned or crossed phase/envelope partitions."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, lfilter
from scipy.stats import spearmanr


def partitions(aligned):
    phase=np.repeat(np.arange(4),8)
    envelope=phase.copy() if aligned else np.tile(np.repeat(np.arange(4),2),4)
    return phase,envelope


def stationary_ar(rng,t,m,rho):
    innovation=rng.normal(size=(t,m))*np.sqrt(1-rho*rho)
    initial=rng.normal(size=(1,m))*rho
    return lfilter([1.],[1.,-rho],innovation,axis=0,zi=initial)[0]


def simulate(aligned,seed,t=1000,*,ranges=None):
    rng=np.random.default_rng(seed);n=32;phase_group,amp_group=partitions(aligned)
    ranges={} if ranges is None else ranges
    if set(ranges)-{'carrier','phase_sd','amp_rho'}:raise ValueError('unknown regime parameter')
    f=rng.uniform(*ranges.get('carrier',(6,12)))
    phase_sd=rng.uniform(*ranges.get('phase_sd',(.10,.18)))
    rho=rng.uniform(*ranges.get('amp_rho',(.94,.98)))
    amp_sd=rng.uniform(.25,.55);jitter=rng.uniform(.15,.35);noise=rng.uniform(.05,.15)
    theta=rng.uniform(-np.pi,np.pi,4)+np.cumsum(2*np.pi*f/100+phase_sd*rng.normal(size=(t,4)),axis=0)
    global_amp=stationary_ar(rng,t,4,rho);local_amp=stationary_ar(rng,t,n,rho)
    logamp=amp_sd*(np.sqrt(.8)*global_amp[:,amp_group]+np.sqrt(.2)*local_amp)
    offsets=rng.uniform(-np.pi,np.pi,n);gains=np.exp(rng.uniform(-.3,.3,n))
    phase=theta[:,phase_group]+offsets+jitter*rng.normal(size=(t,n))
    x=gains*(np.exp(logamp-.5*amp_sd**2)*np.cos(phase)+noise*rng.normal(size=(t,n)))
    order=rng.permutation(n)
    return x[:,order].astype(np.float32),dict(carrier=f,phase_sd=phase_sd,amp_rho=rho,
        amp_sd=amp_sd,phase_jitter=jitter,noise=noise,order=order.tolist(),
        phase_group=phase_group[order].tolist(),envelope_group=amp_group[order].tolist())


def summarize(x):
    return np.r_[np.mean(x),np.std(x),np.quantile(x,[.1,.25,.5,.75,.9])]


def observed_controls(x):
    x=np.asarray(x,dtype=float);filtered=sosfiltfilt(butter(4,[3,20],fs=100,btype='bandpass',output='sos'),x,axis=0)
    analytic=hilbert(filtered,axis=0);trim=min(100,len(x)//10);analytic=analytic[trim:-trim]
    unit=analytic/np.maximum(abs(analytic),1e-12)
    phase=abs(unit.conj().T@unit/len(unit))
    envelope=np.corrcoef(np.log(np.maximum(abs(analytic),1e-12)).T)
    mask=~np.eye(x.shape[1],dtype=bool);p=phase[mask];a=envelope[mask]
    agreement=np.array([np.corrcoef(p,a)[0,1],spearmanr(p,a).statistic])
    return dict(phase_summary=summarize(p),envelope_summary=summarize(a),
                direct_agreement=agreement,phase_matrix=phase,envelope_matrix=envelope)
